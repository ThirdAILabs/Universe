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

  void setWeightGradients(const float* update_weight_gradient) final;
  
  void setBiasesGradients(const float* update_bias_gradient) final;

  float* getBiasesGradient() final;
  
  float* getWeightsGradient() final;

  bool isShallow() final { return _is_shallow; }

  void setShallow(bool shallow) final;

  void setShallowSave(bool shallow) final;
  void buildLayerSummary(std::stringstream& summary, bool detailed) override;

  ~FullyConnectedLayer() = default;

 private:
  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  bool _is_shallow, _shallow_save;
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
  std::unique_ptr<hashing::HashFunction> _hasher;
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

  static std::unique_ptr<hashing::HashFunction> assignHashFunction(
      const SamplingConfig& config, uint64_t dim) {
    switch (config._hash_function) {
      case HashFunctionEnum::DWTA:
        return std::make_unique<hashing::DWTAHashFunction>(
            dim, config.hashes_per_table, config.num_tables, config.range_pow);

      case HashFunctionEnum::FastSRP:
        return std::make_unique<hashing::FastSRP>(dim, config.hashes_per_table,
                                                  config.num_tables);

      case HashFunctionEnum::SRP:
        return std::make_unique<hashing::SparseRandomProjection>(
            dim, config.hashes_per_table, config.num_tables);

      // Not supposed to reach here but compiler complains
      default:
        throw std::invalid_argument("Hash function not supported.");
    }
  }

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
    if (_is_shallow || _shallow_save) {
      archive(true);
      archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
              _biases, _sampling_config, _prev_is_active, _is_active, _hasher,
              _hash_table, _rand_neurons, _force_sparse_for_inference);
    } else {
      archive(false);
      archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
              _biases, _sampling_config, _prev_is_active, _is_active, _hasher,
              _hash_table, _rand_neurons, _force_sparse_for_inference,
              _w_gradient, _w_momentum, _w_velocity, _b_gradient, _b_momentum,
              _b_velocity);
    }
  }

  /**
   * Load first whether the layer is shallow
   * Does not load the optimizer state if is_shallow
   * Loads the optimizer state if !is_shallow
   */
  template <class Archive>
  void load(Archive& archive) {
    archive(_is_shallow);
    if (_is_shallow) {
      archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
              _biases, _sampling_config, _prev_is_active, _is_active, _hasher,
              _hash_table, _rand_neurons, _force_sparse_for_inference);
    } else {
      archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
              _biases, _sampling_config, _prev_is_active, _is_active, _hasher,
              _hash_table, _rand_neurons, _force_sparse_for_inference,
              _w_gradient, _w_momentum, _w_velocity, _b_gradient, _b_momentum,
              _b_velocity);
    }
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
