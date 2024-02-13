#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include "LayerUtils.h"
#include <bolt/src/neuron_index/NeuronIndex.h>
#include <bolt/src/nn/optimizers/Adam.h>
#include <bolt/src/nn/optimizers/Optimizer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DWTA.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <archive/src/Archive.h>
#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>

namespace thirdai::bolt {

class FullyConnected;
class PatchEmbedding;

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnectedLayer final {
  friend class PatchEmbedding;
  friend class FullyConnected;
  friend class tests::FullyConnectedLayerTestFixture;

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

  explicit FullyConnectedLayer(const ar::Archive& archive);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, size_t train_steps);

  void disableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = true;
  };

  void enableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = false;
  };

  void saveWithOptimizer(bool should_save_optimizer) {
    _should_serialize_optimizer = should_save_optimizer;
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

  float* getWeightGradientsPtr() { return _weight_gradients.data(); }

  float* getBiasGradientsPtr() { return _bias_gradients.data(); }

  bool hasOptimizers() const { return _weight_optimizer && _bias_optimizer; }

  std::vector<float>& weightsGradient() { return _weight_gradients; }

  std::vector<float>& biasGradient() { return _bias_gradients; }

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

  void buildLayerSummary(std::stringstream& summary, bool detailed) const;

  void buildSamplingSummary(std::ostream& summary) const;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory);

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;

  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _biases;

  std::vector<float> _weight_gradients;
  std::vector<float> _bias_gradients;

  OptimizerPtr _weight_optimizer;
  OptimizerPtr _bias_optimizer;
  bool _should_serialize_optimizer;

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
  void serialize(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _neuron_index, _index_frozen,
            _disable_sparse_parameter_updates, _use_bias,
            _should_serialize_optimizer);

    if (_should_serialize_optimizer &&
        std::is_same_v<Archive, cereal::BinaryInputArchive>) {
      AdamOptimizer weight_optimizer;
      AdamOptimizer bias_optimizer;

      archive(weight_optimizer, bias_optimizer);

      _weight_optimizer =
          Adam::fromOldOptimizer(std::move(weight_optimizer), _dim, _prev_dim);

      _bias_optimizer =
          Adam::fromOldOptimizer(std::move(bias_optimizer), _dim, 1);

      _weight_gradients.assign(_weights.size(), 0.0);
      _bias_gradients.assign(_biases.size(), 0.0);
    }

    initActiveNeuronsTrackers();
  }
};

}  // namespace thirdai::bolt