#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include <bolt/src/layers/Optimizer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>

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

  void checkpointInMemory();

  void loadCheckpointFromMemory();

  float getSparsity() const { return _sparsity; }

  void setSparsity(float sparsity);

  ActivationFunction getActivationFunction() const { return _act_func; }

  void buildLayerSummary(std::stringstream& summary, bool detailed) const;

  void initOptimizer();

  ~FullyConnectedLayer() = default;

 private:
  template <bool DENSE>
  constexpr uint32_t nonzerosInOutput() const {
    if constexpr (DENSE) {
      return _dim;
    } else {
      return _sparse_dim;
    }
  }

  inline void updateSparseSparseWeightParameters(float lr, float B1, float B2,
                                                 float eps,
                                                 float B1_bias_corrected,
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

  /**
   * If force_build=true build hash tables, return if false.
   * For non-trainable layers, buildHashTablesImpl is called with
   * force_build=false except during initialization and setting weights.
   * For trainable layers, buildHashTablesImpl is always called with
   * force_build=true.
   */
  void buildHashTablesImpl(bool force_build);

  void randomNeuronSampling(const BoltVector& input, const BoltVector& output,
                            const BoltVector* labels);

  template <bool PREV_DENSE>
  void lshNeuronSampling(const BoltVector& input, BoltVector& output,
                         const BoltVector* labels);

  bool useRandomSampling() const {
    return _sampling_mode == BoltSamplingMode::RandomSampling;
  }

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _hasher, _hash_table, _rand_neurons, _sampling_mode,
            _prev_is_active, _is_active);
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
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _hasher, _hash_table, _rand_neurons, _sampling_mode,
            _prev_is_active, _is_active);
  }

  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _trainable;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _biases;

  struct WeightBiasCheckpoint {
    explicit WeightBiasCheckpoint(std::vector<float> _weights,
                                  std::vector<float> _biases)
        : weights(_weights), biases(_biases) {}

    std::vector<float> weights;
    std::vector<float> biases;
  };

  std::optional<WeightBiasCheckpoint> _weight_bias_checkpoint = std::nullopt;

  std::optional<AdamOptimizer> _weight_optimizer = std::nullopt;
  std::optional<AdamOptimizer> _bias_optimizer = std::nullopt;

  std::unique_ptr<hashing::HashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  // This set of variables are only used within a batch, so we do not use
  // _this_is_dense to check if the layer is sparse, we instead check if
  // _sparsity is less than 1.
  bool _prev_is_dense;
  bool _this_is_dense;

  using ActiveNeuronsPair =
      std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;

  // This is only used if _prev_is_dense == false and _this_is_dense == false
  // This is a vector of unique_ptr so that the push_back in the critical
  // region is just a pointer move and can be very fast
  std::vector<std::unique_ptr<ActiveNeuronsPair>> _active_pairs;
  // This is only used if _prev_is_dense == false
  std::vector<bool> _prev_is_active;
  // This is only used if _this_is_dense == false
  std::vector<bool> _is_active;

  // A flag to check whether the current network is running in the normal
  // settings and distributed settings
  bool _is_distributed;

  BoltSamplingMode _sampling_mode;
};

}  // namespace thirdai::bolt
