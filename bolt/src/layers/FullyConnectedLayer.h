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

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnectedLayer final : public SequentialLayer {
  friend class tests::FullyConnectedLayerTestFixture;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim);

  void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels) final;

  void backpropagate(BoltVector& input, BoltVector& output) final;

  void backpropagateInputLayer(BoltVector& input, BoltVector& output) final;

  void updateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) final;

  BoltBatch createBatchState(const uint32_t batch_size,
                             bool force_dense) const final {
    bool is_dense = (_sparse_dim == _dim) || force_dense;

    return BoltBatch(is_dense ? _dim : _sparse_dim, batch_size, is_dense);
  }

  void forceSparseForInference() final {
    if (_sparsity < 1.0) {
      _force_sparse_for_inference = true;
    }
  }

  bool isForceSparsity() const final { return _force_sparse_for_inference; }

  void buildHashTables() final;

  void reBuildHashFunction() final;

  void shuffleRandNeurons() final;

  uint32_t getDim() const final { return _dim; }

  uint32_t getInputDim() const final { return _prev_dim; }

  uint32_t getInferenceOutputDim() const final {
    if (_force_sparse_for_inference) {
      return _sparse_dim;
    }
    return _dim;
  }

  float* getWeights() final;

  float* getBiases() final;

  void setTrainable(bool trainable) final;

  bool getTrainable() final;

  void setWeights(const float* new_weights) final;

  void setBiases(const float* new_biases) final;

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _trainable, _force_build;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _w_gradient;
  std::vector<float> _w_momentum;
  std::vector<float> _w_velocity;

  std::vector<float> _biases;
  std::vector<float> _b_gradient;
  std::vector<float> _b_momentum;
  std::vector<float> _b_velocity;

  SamplingConfig _sampling_config;
  std::unique_ptr<hashing::DWTAHashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  using ActiveNeuronsPair =
      std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;

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

  bool _force_sparse_for_inference;

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
  void forwardImpl(const BoltVector& input, BoltVector& output,
                   const BoltVector* labels);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const BoltVector& input, BoltVector& output,
                           const BoltVector* labels);

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _w_gradient, _w_momentum, _w_velocity, _biases, _b_gradient,
            _b_momentum, _b_velocity, _sampling_config, _prev_is_active,
            _is_active, _hasher, _hash_table, _rand_neurons,
            _force_sparse_for_inference);
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

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedLayer)
CEREAL_REGISTER_POLYMORPHIC_RELATION(thirdai::bolt::SequentialLayer,
                                     thirdai::bolt::FullyConnectedLayer)
