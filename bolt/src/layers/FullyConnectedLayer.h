#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include "cereal/types/utility.hpp"
#include <bolt/src/layers/Optimizer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>
#include <optional>
#include <random>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

enum class BoltSamplingMode {
  LSH,
  FreezeHashTables,
  FreezeHashTablesWithInsertions,
  RandomSampling
};

class FullyConnectedLayer final {
  friend class tests::FullyConnectedLayerTestFixture;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim, bool is_distributed = false);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  void enableDistributedTraining() { _is_distributed = true; };

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool use_sparsity) const {
    bool is_sparse = (_sparsity < 1.0) && use_sparsity;

    uint32_t curr_dim = is_sparse ? _sparse_dim : _dim;

    return BoltBatch(/* dim= */ curr_dim, /* batch_size= */ batch_size,
                     /* is_dense= */ !is_sparse);
  }

  void freezeHashTables(bool insert_labels_if_not_found) {
    if (useRandomSampling()) {
      return;
    }

    if (insert_labels_if_not_found) {
      _sampling_mode = BoltSamplingMode::FreezeHashTablesWithInsertions;
    } else {
      _sampling_mode = BoltSamplingMode::FreezeHashTables;
    }
  }

  bool hashTablesFrozen() const {
    return _sampling_mode == BoltSamplingMode::FreezeHashTables ||
           _sampling_mode == BoltSamplingMode::FreezeHashTablesWithInsertions;
  }

  void buildHashTables();

  void reBuildHashFunction();

  uint32_t getDim() const { return _dim; }

  uint32_t getInputDim() const { return _prev_dim; }

  uint32_t getSparseDim() const { return _sparse_dim; }

  float* getWeightsPtr() { return _weights.data(); }

  float* getBiasesPtr() { return _biases.data(); }

  float* getWeightGradientsPtr() { return _weight_optimizer->gradients.data(); }

  float* getBiasGradientsPtr() { return _bias_optimizer->gradients.data(); }

  float* getWeights() const;

  float* getBiases() const;

  void setTrainable(bool trainable);

  bool getTrainable() const;

  void setWeights(const float* new_weights);

  void setBiases(const float* new_biases);

  void setWeightGradients(const float* update_weight_gradient);

  void setBiasesGradients(const float* update_bias_gradient);

  float* getBiasesGradient();

  float* getWeightsGradient();

  float getSparsity() const { return _sparsity; }

  void setSparsity(float sparsity);

  ActivationFunction getActivationFunction() const { return _act_func; }

  void buildLayerSummary(std::stringstream& summary, bool detailed) const;

  void initOptimizer();

  void enableSparseSparseOptimization() {
    _use_sparse_sparse_optimization = true;
  }

  void disableSparseSparseOptimization() {
    _use_sparse_sparse_optimization = false;
  }

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _trainable;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _biases;

  std::optional<AdamOptimizer> _weight_optimizer = std::nullopt;
  std::optional<AdamOptimizer> _bias_optimizer = std::nullopt;

  std::unique_ptr<hashing::HashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  template <bool DENSE>
  constexpr uint32_t nonzerosInOutput() const {
    if constexpr (DENSE) {
      return _dim;
    } else {
      return _sparse_dim;
    }
  }

  // A flag to check whether the current network is running in normal
  // or distributed mode
  bool _is_distributed;

  BoltSamplingMode _sampling_mode;

  // Whether to track ActivePairs in raw form (the sparse sparse "optimization")
  // Default set in constructor is false
  bool _use_sparse_sparse_optimization;

  // --------------- Within-batch variables ------------------------------
  // These variables are set while we are processing a batch (usually during
  // calls to forward) and are used later, usually while in updateParameters.
  // Since these are temporary within batch variables, we do NOT
  // serialize them, and we reinitialize them upon deserialization (so they
  // have the correct size).
  // Overall, these variables are
  //  - initialized during the constructor/deserialization
  //  - populated during forward
  //  - used during updateParameters
  //  - cleaned up/zero'd during updateParameters (with a call to
  //    cleanupWithinBatchVariables)

  // These track whether the current/previous layer was dense (using whether
  // the BoltVectors in forward are dense).
  bool _prev_is_dense;
  bool _this_is_dense;

  // The following variables track which neurons were active during batch
  // training.

  // _prev_is_active is only used if _prev_is_dense == false and _cur_is_dense
  // is true.
  // It tracks the neurons in the previous layer which are active.
  std::vector<bool> _prev_is_active;

  // _is_active is used if _this_is_dense == false.
  // It tracks the neurons in the current layer which are active.
  std::vector<bool> _is_active;

  /* The following two variables are only used if _prev_is_dense == false and
   * _this_is_dense == false. They track exactly which pairs of neurons were
   * active in two different ways: _active_pairs_array marks position
   * cur_neuron * _prev_dim + prev_neuron with true if (prev_neuron, cur_neuron)
   * was active during the training of the current batch. _active_pairs_raw
   * appends an ActivePair object for each example. The ActivePair object
   * contains all of the [prev_neurons] and [cur_neurons] that were active for
   * that example (the active pairs for that example are then a cartesian
   * product of those two lists).
   * IMPORTANT: we only use _active_pairs_raw when
   * _use_sparse_sparse_optimization is true.
   */
  std::vector<bool> _active_pairs_array;
  using ActiveNeuronsPair =
      std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;
  std::vector<std::unique_ptr<ActiveNeuronsPair>> _active_pairs_raw;

  // -------------------------------------------------------------------------

  void initActiveNeuronsTrackers();

  bool useRandomSampling() const {
    return _sampling_mode == BoltSamplingMode::RandomSampling;
  }

  inline void updateSparseSparseWeightParametersNormal(float lr, float B1,
                                                       float B2, float eps,
                                                       float B1_bias_corrected,
                                                       float B2_bias_corrected);
  inline void updateSparseSparseWeightParametersOptimized(
      float lr, float B1, float B2, float eps, float B1_bias_corrected,
      float B2_bias_corrected);
  inline void updateSparseDenseWeightParameters(float lr, float B1, float B2,
                                                float eps,
                                                float B1_bias_corrected,
                                                float B2_bias_corrected);
  inline void updateDenseSparseWeightParameters(float lr, float B1, float B2,
                                                float eps,
                                                float B1_bias_corrected,
                                                float B2_bias_corrected);
  inline void updateDenseDenseWeightParameters(float lr, float B1, float B2,
                                               float eps,
                                               float B1_bias_corrected,
                                               float B2_bias_corrected);
  inline void updateSingleWeightParameters(uint64_t prev_neuron,
                                           uint64_t cur_neuron, float lr,
                                           float B1, float B2, float eps,
                                           float B1_bias_corrected,
                                           float B2_bias_corrected);
  inline void updateBiasParameters(float lr, float B1, float B2, float eps,
                                   float B1_bias_corrected,
                                   float B2_bias_corrected);

  inline void cleanupWithinBatchVars();

  inline void initSamplingDatastructures(
      const SamplingConfigPtr& sampling_config, std::random_device& rd);

  inline void deinitSamplingDatastructures();

  template <bool DENSE, bool PREV_DENSE>
  void markActiveNeuronsForUpdate(const BoltVector& input,
                                  const BoltVector& output, uint32_t len_out);

  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output,
                   const BoltVector* labels);

  void eigenDenseDenseForward(const BoltVector& input, BoltVector& output);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  template <bool FIRST_LAYER>
  void eigenDenseDenseBackpropagate(BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const BoltVector& input, BoltVector& output,
                           const BoltVector* labels);

  void randomNeuronSampling(const BoltVector& input, const BoltVector& output,
                            const BoltVector* labels);

  template <bool PREV_DENSE>
  void lshNeuronSampling(const BoltVector& input, BoltVector& output,
                         const BoltVector* labels);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _trainable, _act_func,
            _weights, _biases, _hasher, _hash_table, _rand_neurons,
            _is_distributed, _sampling_mode, _use_sparse_sparse_optimization);
  }

  /**
   * Training data-structures (like the optimizer and the active neurons
   * trackers) are not loaded in by default. If we want to continue training
   * after a load, the expectation is that the higher level Graph/Network API
   * will handle this initialization with the initOptimizer() method.
   *
   * Doing this means our load API is as simple as possible for both
   * training and inference purposes. It doesn't make sense to load these
   * data-structures by default then remove them with another function since
   * users may be memory constrained during deployment.
   *
   * We don't know yet if its worth it to save the optimizer for
   * retraining/finetuning purposes. If in the future we figure out this has
   * some benefit we can adjust this method accordingly.
   */
  template <class Archive>
  void load(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _trainable, _act_func,
            _weights, _biases, _hasher, _hash_table, _rand_neurons,
            _is_distributed, _sampling_mode, _use_sparse_sparse_optimization);

    /**
     * Here we init the optimizer so that any calls to train in the network
     * are safe. If we need to reduce memory usage for smaller machines we can
     * use the removeOptimizer() method to remove these parameters. This will
     * also likely require adding an additional node state for uninitialized
     * optimizers so that we have memory safety.
     */
    initOptimizer();

    // TODO(david) another way to reduce memory for inference is to remove these
    // in addition to the optimizer as mentioned above
    initActiveNeuronsTrackers();
  }

  /**
   * If force_build=true build hash tables, return if false.
   * For non-trainable layers, buildHashTablesImpl is called with
   * force_build=false except during initialization and setting weights.
   * For trainable layers, buildHashTablesImpl is always called with
   * force_build=true.
   */
  void buildHashTablesImpl(bool force_build);
};

}  // namespace thirdai::bolt
