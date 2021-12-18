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

class FullyConnectedLayer final : public Layer {
  friend class tests::FullyConnectedLayerTestFixture;

 public:
  FullyConnectedLayer() {}

  FullyConnectedLayer(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer(FullyConnectedLayer&&) = delete;
  FullyConnectedLayer& operator=(const FullyConnectedLayer&) = delete;
  FullyConnectedLayer& operator=(FullyConnectedLayer&&) = delete;

  FullyConnectedLayer(const FullyConnectedLayerConfig& config,
                      uint64_t prev_dim);

  void feedForward(uint32_t batch_indx, const uint32_t* indices,
                   const float* values, uint32_t len, uint32_t* labels,
                   uint32_t label_len) override;

  void backpropagate(uint32_t batch_indx, const uint32_t* indices,
                     const float* values, float* errors, uint32_t len) override;

  void backpropagateFirstLayer(uint32_t batch_indx, const uint32_t* indices,
                               const float* values, uint32_t len) override;

  void computeSoftmaxErrors(uint32_t batch_indx, uint32_t batch_size,
                            const uint32_t* labels,
                            uint32_t label_len) override;

  void computeMeanSquaredErrors(uint32_t batch_indx, uint32_t batch_size,
                                const uint32_t* truth_indices,
                                const float* truth_values,
                                uint32_t truth_len) override;

  void updateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) override;

  void buildHashTables() override;

  void reBuildHashFunction() override;

  void setSparsity(float new_sparsity) override;

  void initializeLayer(uint64_t new_batch_size) override;

  void initializeLayer(uint64_t new_batch_size, float** new_activations,
                       float** new_errors);

  void shuffleRandNeurons() override;

  uint32_t getLen(uint32_t batch_indx) const override {
    return _active_lens[batch_indx];
  }

  const uint32_t* getIndices(uint32_t batch_indx) const override {
    return _active_neurons[batch_indx];
  }

  const float* getValues(uint32_t batch_indx) const override {
    return _activations[batch_indx];
  }

  float* getErrors(uint32_t batch_indx) override { return _errors[batch_indx]; }

  float* getWeights();

  float* getBiases();

  ~FullyConnectedLayer();

 private:
  template <bool DENSE, bool PREV_DENSE>
  void feedForwardImpl(uint32_t batch_indx, const uint32_t* indices,
                       const float* values, uint32_t len, uint32_t* labels,
                       uint32_t label_len);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void backPropagateImpl(uint32_t batch_indx, const uint32_t* indices,
                         const float* values, float* errors, uint32_t len);

  template <bool DENSE>
  void computeSoftmaxErrorsImpl(uint32_t batch_indx, uint32_t batch_size,
                                const uint32_t* labels, uint32_t label_len);

  template <bool DENSE, bool TRUTH_DENSE>
  void computeMeanSquaredErrorsImpl(uint32_t batch_indx, uint32_t batch_size,
                                    const uint32_t* truth_indices,
                                    const float* truth_values,
                                    uint32_t truth_len);

  template <bool DENSE, bool PREV_DENSE>
  void selectActiveNeurons(uint32_t batch_indx, const uint32_t* indices,
                           const float* values, uint32_t len, uint32_t* labels,
                           uint32_t label_len);

  constexpr float actFuncDerivative(float x);

  void deallocateInternalState();

  uint64_t _dim, _prev_dim, _max_batch_size, _sparse_dim;
  float _sparsity;
  ActivationFunc _act_func;

  uint32_t* _active_lens;
  uint32_t** _active_neurons;
  float** _activations;
  float** _errors;

  bool _internal_state_provided;

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