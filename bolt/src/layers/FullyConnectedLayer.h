#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "BoltVector.h"
#include "LayerConfig.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnectedLayer {
  friend class tests::FullyConnectedLayerTestFixture;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim);

  virtual void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels = nullptr);

  virtual void backpropagate(BoltVector& input, BoltVector& output);

  virtual void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  virtual void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  virtual BoltBatch createBatchState(const uint32_t batch_size,
                             bool force_dense = false) const {
    bool is_dense = (_sparse_dim == _dim) || force_dense;

    return BoltBatch(is_dense ? _dim : _sparse_dim, batch_size, is_dense);
  }

  void forceSparseForInference() {
    if (_sparsity < 1.0) {
      _force_sparse_for_inference = true;
    }
  }

  bool isForceSparsity() const { return _force_sparse_for_inference; }

  virtual void buildHashTables();

  virtual void reBuildHashFunction();

  void shuffleRandNeurons();

  uint32_t getDim() const { return _dim; }

  float* getWeights();

  float* getBiases();

  void setWeights(float* new_weights);

  void setBiases(float* new_biases);

  virtual ~FullyConnectedLayer() = default;

 protected:
  constexpr float actFuncDerivative(float x) {
    switch (_act_func) {
      case ActivationFunction::ReLU:
        return x > 0 ? 1.0 : 0.0;
      case ActivationFunction::Softmax:
        // return 1.0; // Commented out because Clang tidy doesn't like
        // consecutive identical branches
      case ActivationFunction::Linear:
        return 1.0;
        // default:
        //   return 0.0;
    }
    // This is impossible to reach, but the compiler gave a warning saying it
    // reached the end of a non void function without it.
    return 0.0;
  }

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output,
                   const BoltVector* labels);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const BoltVector& input, BoltVector& output,
                           const BoltVector* labels);

  uint64_t _dim, _prev_dim, _sparse_dim;
  float _sparsity;
  ActivationFunction _act_func;

  std::vector<float> _weights;
  std::vector<float> _w_gradient;
  std::vector<float> _w_momentum;
  std::vector<float> _w_velocity;

  std::vector<float> _biases;
  std::vector<float> _b_gradient;
  std::vector<float> _b_momentum;
  std::vector<float> _b_velocity;

  std::vector<bool> _is_active;

  SamplingConfig _sampling_config;
  std::unique_ptr<hashing::DWTAHashFunction> _hasher;
  std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
  std::vector<uint32_t> _rand_neurons;

  bool _force_sparse_for_inference;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_dim, _prev_dim, _sparse_dim, _sparsity, _act_func, _weights,
            _w_gradient, _w_momentum, _w_velocity, _biases, _b_gradient,
            _b_momentum, _b_velocity, _is_active, _sampling_config, _hasher,
            _hash_table, _rand_neurons, _force_sparse_for_inference);
  }
};

}  // namespace thirdai::bolt