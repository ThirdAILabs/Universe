#pragma once

#include "../../utils/hashing/DWTA.h"
#include "../../utils/hashtable/SampledHashTable.h"
#include "Layer.h"
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

  FullyConnectedLayer(uint64_t dim, uint64_t prev_dim, float sparsity,
                      ActivationFunc act_func, SamplingConfig sampling_config);

  void FeedForward(uint32_t batch_indx, const uint32_t* indices,
                   const float* values, uint32_t len, uint32_t* labels,
                   uint32_t label_len) override;

  void Backpropagate(uint32_t batch_indx, const uint32_t* indices,
                     const float* values, float* errors, uint32_t len) override;

  void BackpropagateFirstLayer(uint32_t batch_indx, const uint32_t* indices,
                               const float* values, float* errors,
                               uint32_t len) override;

  void ComputeErrors(uint32_t batch_indx, const uint32_t* labels,
                     uint32_t label_len) override;

  void UpdateParameters(float lr, uint32_t iter, float B1, float B2,
                        float eps) override;

  void BuildHashTables() override;

  void ReBuildHashFunction() override;

  void SetSparsity(float new_sparsity) override;

  void SetBatchSize(uint64_t new_batch_size) override;

  void ShuffleRandNeurons() override;

  uint32_t GetLen(uint32_t batch_indx) const override {
    return _active_lens[batch_indx];
  }

  const uint32_t* GetIndices(uint32_t batch_indx) const override {
    return _active_neurons[batch_indx];
  }

  const float* GetValues(uint32_t batch_indx) const override {
    return _activations[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) override { return _errors[batch_indx]; }

  float* GetWeights();

  float* GetBiases();

  ~FullyConnectedLayer();

 private:
  template <bool DENSE, bool PREV_DENSE>
  void FeedForwardImpl(uint32_t batch_indx, const uint32_t* indices,
                       const float* values, uint32_t len, uint32_t* labels,
                       uint32_t label_len);

  template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
  void BackPropagateImpl(uint32_t batch_indx, const uint32_t* indices,
                         const float* values, float* errors, uint32_t len);

  template <bool DENSE>
  void ComputeErrorsImpl(uint32_t batch_indx, const uint32_t* labels,
                         uint32_t label_len);

  template <bool DENSE, bool PREV_DENSE>
  void SelectActiveNeurons(uint32_t batch_indx, const uint32_t* indices,
                           const float* values, uint32_t len, uint32_t* labels,
                           uint32_t label_len);

  constexpr float ActFuncDerivative(float x);

  uint64_t _dim, _prev_dim, _batch_size, _sparse_dim;
  float _sparsity;
  ActivationFunc _act_func;

  uint32_t* _active_lens;
  uint32_t** _active_neurons;
  float** _activations;
  float** _errors;

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
  utils::DWTAHashFunction* _hasher;
  utils::SampledHashTable<uint32_t>* _hash_table;
  uint32_t* _rand_neurons;
};

}  // namespace thirdai::bolt
