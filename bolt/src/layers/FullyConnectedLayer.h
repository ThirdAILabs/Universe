#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "BoltVector.h"
#include "LayerConfig.h"
#include "LayerUtils.h"
#include "SequentialLayer.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>
#include <random>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

enum class LSHSamplingMode {
  Default,
  FreezeHashTables,
  FreezeHashTablesWithInsertions
};

class FullyConnectedLayer final : public SequentialLayer {
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
               const BoltVector* labels) final;

  void backpropagate(BoltVector& input, BoltVector& output) final;

  void backpropagateInputLayer(BoltVector& input, BoltVector& output) final;

  void updateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) final;

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool use_sparsity) const final {
    bool is_sparse = (_sparsity < 1.0) && use_sparsity;

    uint32_t curr_dim = is_sparse ? _sparse_dim : _dim;

    return BoltBatch(/* dim= */ curr_dim, /* batch_size= */ batch_size,
                     /* is_dense= */ !is_sparse);
  }

  void freezeHashTables(bool insert_labels_if_not_found) final {
    if (insert_labels_if_not_found) {
      _sampling_mode = LSHSamplingMode::FreezeHashTablesWithInsertions;
    } else {
      _sampling_mode = LSHSamplingMode::FreezeHashTables;
    }
  }

  bool hashTablesFrozen() const {
    return _sampling_mode == LSHSamplingMode::FreezeHashTables ||
           _sampling_mode == LSHSamplingMode::FreezeHashTablesWithInsertions;
  }

  void buildHashTables() final;

  void reBuildHashFunction() final;

  uint32_t getDim() const final { return _dim; }

  uint32_t getInputDim() const final { return _prev_dim; }

  uint32_t getSparseDim() const final { return _sparse_dim; }

  float* getWeights() const final;

  float* getBiases() const final;

  void setTrainable(bool trainable) final;

  bool getTrainable() const final;

  void setWeights(const float* new_weights) final;

  void setBiases(const float* new_biases) final;

  void setWeightGradients(const float* update_weight_gradient) final;

  void setBiasesGradients(const float* update_bias_gradient) final;

  float* getBiasesGradient() final;

  float* getWeightsGradient() final;

  float getSparsity() const final { return _sparsity; }

  void setSparsity(float sparsity) final;

  ActivationFunction getActivationFunction() const { return _act_func; }

  void buildLayerSummary(std::stringstream& summary,
                         bool detailed) const override;

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _trainable;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _w_gradient;
  std::vector<float> _w_momentum;
  std::vector<float> _w_velocity;

  std::vector<float> _biases;
  std::vector<float> _b_gradient;
  std::vector<float> _b_momentum;
  std::vector<float> _b_velocity;

  std::unique_ptr<hashing::HashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  using ActiveNeuronsPair =
      std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;

  // This set of variables are only used within a batch, so we do not use
  // _this_is_dense to check if the layer is sparse, we instead check if
  // _sparsity is less than 1.
  bool _prev_is_dense;
  bool _this_is_dense;
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

  LSHSamplingMode _sampling_mode;

  void initOptimizer();

  void removeOptimizer();

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

  inline void initSparseDatastructures(const SamplingConfigPtr& sampling_config,
                                       std::random_device& rd);

  inline void deinitSparseDatastructures();

  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output,
                   const BoltVector* labels);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const BoltVector& input, BoltVector& output,
                           const BoltVector* labels);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;

  /**
   * Not serializing _shallow_save because it is only used to decide to how to
   * save the model. If _shallow_save or _is_shallow is true, archive
   * _is_shallow as true. If both are false, archive _is_shallow as false. While
   * dearchiving, we only need to know whether or not the layer is shallow,
   * hence, _shallow_save not archived.
   */
  template <class Archive>
  void save(Archive& archive) const {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _hasher, _hash_table, _rand_neurons, _sampling_mode,
            _prev_is_active, _is_active);
  }

  /**
   * Load first whether the layer is shallow
   * Does not load the optimizer state if is_shallow
   * Loads the optimizer state if !is_shallow
   */
  template <class Archive>
  void load(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _biases, _hasher, _hash_table, _rand_neurons, _sampling_mode,
            _prev_is_active, _is_active);

    /**
     * Here we init the optimizer so that any calls to train in the network are
     * safe. If we need to reduce memory usage for smaller machines we can use
     * the removeOptimizer() method to remove these parameters. This will also
     * likely require adding an additional node state for uninitialized
     * optimizers so that we have memory safety.
     */
    initOptimizer();
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

#include "FullyConnectedLayer.impl"

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(thirdai::bolt::SequentialLayer,
                                     thirdai::bolt::FullyConnectedLayer)
