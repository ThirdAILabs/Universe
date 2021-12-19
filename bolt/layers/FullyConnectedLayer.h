#pragma once

#include "Layer.h"
#include "LayerConfig.h"
#include <hashing/src/DWTA.h>
#include <hashtable/src/SampledHashTable.h>
#include <cstdint>

namespace thirdai::bolt {

namespace tests {
class FullyConnectedLayerTestFixture;
}  // namespace tests

class FullyConnectedLayer final{
  friend class tests::FullyConnectedLayerTestFixture;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim);

  void forward(const VectorState& input, VectorState& output,
               const uint32_t* labels = nullptr,
               uint32_t label_len = 0) ;

  void backpropagate(VectorState& input, VectorState& output) ;

  void backpropagateInputLayer(VectorState& input,
                               VectorState& output) ;

  // void computeSoftmaxErrors(uint32_t batch_indx, uint32_t batch_size,
  //                           const uint32_t* labels,
  //                           uint32_t label_len) override;

  // void computeMeanSquaredErrors(uint32_t batch_indx, uint32_t batch_size,
  //                               const uint32_t* truth_indices,
  //                               const float* truth_values,
  //                               uint32_t truth_len) override;

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  BatchState createBatchState(const uint32_t batch_size,
                              bool force_dense = false)  {
    bool is_dense = (_sparse_dim == _dim) || force_dense;

    return BatchState(is_dense ? _dim : _sparse_dim, batch_size, is_dense);
  }

  void buildHashTables();

  void reBuildHashFunction();

  void shuffleRandNeurons();

  float* getWeights();

  float* getBiases();

  ~FullyConnectedLayer();

 private:
  template <bool DENSE, bool PREV_DENSE>
  void forwardImpl(const VectorState& input, VectorState& output,
                   const uint32_t* labels, uint32_t label_len);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backpropagateImpl(VectorState& input, VectorState& output);

  // template <bool DENSE>
  // void computeSoftmaxErrorsImpl(uint32_t batch_indx, uint32_t batch_size,
  //                               const uint32_t* labels, uint32_t label_len);

  // template <bool DENSE, bool TRUTH_DENSE>
  // void computeMeanSquaredErrorsImpl(uint32_t batch_indx, uint32_t batch_size,
  //                                   const uint32_t* truth_indices,
  //                                   const float* truth_values,
  //                                   uint32_t truth_len);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(const VectorState& input, VectorState& output,
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
};

}  // namespace thirdai::bolt