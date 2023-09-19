#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/neuron_index/NeuronIndex.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DWTA.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <proto/ops.pb.h>
#include <cstdint>
#include <optional>
#include <random>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnected;

class FullyConnectedLayer final {
  friend class tests::FullyConnectedLayerTestFixture;
  friend class FullyConnected;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim,
                      bool disable_sparse_parameter_updates = false,
                      bool use_bias = true);

  explicit FullyConnectedLayer(const proto::bolt::FullyConnected& fc_proto);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  void disableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = true;
  };

  void enableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = false;
  };

  void saveWithOptimizer(bool should_save_optimizer) {
    _should_save_optimizer = should_save_optimizer;
  }

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool use_sparsity) const {
    bool is_sparse = (_sparsity < 1.0) && use_sparsity;

    uint32_t curr_dim = is_sparse ? _sparse_dim : _dim;

    return BoltBatch(/* dim= */ curr_dim, /* batch_size= */ batch_size,
                     /* is_dense= */ !is_sparse);
  }

  void freezeHashTables(bool insert_labels_if_not_found);

  void unfreezeHashTables() { _index_frozen = false; }

  bool isNeuronIndexFrozen() const { return _index_frozen; }

  void buildHashTables();

  void reBuildHashFunction();

  const NeuronIndexPtr& neuronIndex() const { return _neuron_index; }

  void setNeuronIndex(NeuronIndexPtr index);

  uint32_t getDim() const { return _dim; }

  uint32_t getInputDim() const { return _prev_dim; }

  uint32_t getSparseDim() const { return _sparse_dim; }

  float* getWeightsPtr() { return _weights.data(); }

  float* getBiasesPtr() { return _biases.data(); }

  float* getWeightGradientsPtr() { return _weight_optimizer->gradients.data(); }

  float* getBiasGradientsPtr() { return _bias_optimizer->gradients.data(); }

  std::vector<float>& weightsGradient() { return _weight_optimizer->gradients; }

  std::vector<float>& biasGradient() { return _bias_optimizer->gradients; }

  std::vector<float>& weights() { return _weights; }

  std::vector<float>& biases() { return _biases; }

  float* getWeights() const;

  float* getBiases() const;

  bool useBias() const { return _use_bias; }

  void setWeights(const float* new_weights);

  void setBiases(const float* new_biases);

  void setWeightGradients(const float* update_weight_gradient);

  void setBiasesGradients(const float* update_bias_gradient);

  float* getBiasesGradient();

  float* getWeightsGradient();

  std::vector<float> getWeightsByNeuron(uint32_t neuron_id);

  float getSparsity() const { return _sparsity; }

  void setSparsity(float sparsity, bool rebuild_hash_tables,
                   bool experimental_autotune);

  ActivationFunction getActivationFunction() const { return _act_func; }

  std::pair<hashing::HashFunctionPtr, hashtable::SampledHashTablePtr>
  getHashTable();

  void setHashTable(hashing::HashFunctionPtr hash_fn,
                    hashtable::SampledHashTablePtr hash_table);

  proto::bolt::FullyConnected* toProto(bool with_optimizer) const;

  void buildLayerSummary(std::stringstream& summary, bool detailed) const;

  void buildSamplingSummary(std::ostream& summary) const;

  void initOptimizer();

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;

  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _biases;

  std::optional<AdamOptimizer> _weight_optimizer = std::nullopt;
  std::optional<AdamOptimizer> _bias_optimizer = std::nullopt;

  NeuronIndexPtr _neuron_index;
  bool _index_frozen = false;

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
  bool _disable_sparse_parameter_updates;

  // A flag to determine whether the current network saves the optimizer states
  // or not. If true, it saves the optimizer states, else doesn't.
  bool _should_save_optimizer;

  bool _use_bias;

  /* --------------- Within-batch variables ------------------------------
   * These variables are set while we are processing a batch (usually during
   * calls to forward) and are used later, usually while in updateParameters.
   * Since these are temporary within batch variables, we do NOT
   * serialize them, and we reinitialize them upon deserialization (so they
   * have the correct size).
   * Overall, these variables are
   *  - initialized during the constructor/deserialization
   *  - populated during forward
   *  - used during updateParameters
   *  - cleaned up/zero'd during updateParameters (with a call to
   *    cleanupWithinBatchVariables for everything except _active_pairs_array,
   *    which we clean up while we update the weights.)
   */

  // These track whether the current/previous layer was dense (using whether
  // the BoltVectors in forward are dense).
  bool _prev_is_dense;
  bool _this_is_dense;

  // The following variables track which neurons were active during batch
  // training.

  // _prev_is_active is used if _prev_is_dense == false. It tracks the neurons
  // in the previous layer which are active at some point in the batch.
  std::vector<bool> _prev_is_active;

  // _is_active is used if _this_is_dense == false. It tracks the neurons in the
  // current layer which are active at some point in the batch.
  std::vector<bool> _is_active;

  // -------------------------------------------------------------------------

  void initActiveNeuronsTrackers();

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

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _neuron_index, _index_frozen,
            _disable_sparse_parameter_updates, _should_save_optimizer,
            _use_bias);
    if (_should_save_optimizer) {
      archive(_weight_optimizer, _bias_optimizer);
    }
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
            _biases, _neuron_index, _index_frozen,
            _disable_sparse_parameter_updates, _should_save_optimizer,
            _use_bias);
    if (_should_save_optimizer) {
      archive(_weight_optimizer, _bias_optimizer);
    }
    // TODO(david) another way to reduce memory for inference is to remove these
    // in addition to the optimizer as mentioned above
    initActiveNeuronsTrackers();
  }
};

}  // namespace thirdai::bolt