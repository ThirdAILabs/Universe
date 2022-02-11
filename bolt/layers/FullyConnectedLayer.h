#pragma once

#include "BoltVector.h"
#include "LayerConfig.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnectedLayer final {
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
               const uint32_t* labels = nullptr, uint32_t label_len = 0);

  void backpropagate(BoltVector& input, BoltVector& output);

  void backpropagateInputLayer(BoltVector& input, BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  BoltBatch createBatchState(const uint32_t batch_size,
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

  bool isRestrictClass() const { return _is_restricted_class; }

  // Anshu: For Multi-Task
  void restrictClass(uint32_t* class_ids, uint32_t class_ids_len);

  void buildHashTables();

  void reBuildHashFunction();

  void shuffleRandNeurons();

  float* getWeights();

  float* getBiases();

  ~FullyConnectedLayer();

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const BoltVector& input, BoltVector& output,
                   const uint32_t* labels, uint32_t label_len);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(BoltVector& input, BoltVector& output);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const BoltVector& input, BoltVector& output,
                           const uint32_t* labels, uint32_t label_len);

  constexpr float actFuncDerivative(float x);

  uint64_t _dim, _prev_dim, _max_batch_size, _sparse_dim;
  float _sparsity;
  ActivationFunc _act_func;

  float* _weights;
  float* _w_gradient;
  float* _w_momentum;
  float* _w_velocity;

  float* _biases;
  float* _b_gradient;
  float* _b_momentum;
  float* _b_velocity;

  bool* _is_active;

  SamplingConfig _sampling_config;
  hashing::DWTAHashFunction* _hasher;
  hashtable::SampledHashTable<uint32_t>* _hash_table;
  uint32_t* _rand_neurons;

  bool _force_sparse_for_inference;

  // Anshu: For MultiTask
  bool _is_restricted_class;
  uint32_t* _restricted_class;
  uint32_t _restricted_class_len;
};

}  // namespace thirdai::bolt