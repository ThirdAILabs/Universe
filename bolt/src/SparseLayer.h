#pragma once

#include "../../utils/hashing/DWTA.h"
#include "../../utils/hashtable/SampledHashTable.h"
#include "Layer.h"
#include <stdint.h>

namespace thirdai::bolt {

namespace tests {
class SparseLayerTestFixture;
}  // namespace tests

class SparseLayer final : public Layer {
  friend class tests::SparseLayerTestFixture;

 public:
  SparseLayer() {}

  SparseLayer(const SparseLayer&) = delete;
  SparseLayer(SparseLayer&&) = delete;
  SparseLayer& operator=(const SparseLayer&) = delete;
  SparseLayer& operator=(SparseLayer&&) = delete;

  SparseLayer(uint64_t _dim, uint64_t _prev_dim, float _sparsity,
              ActivationFunc _act_func, SamplingConfig _sampling_config);

  void FeedForward(uint32_t batch_indx, const uint32_t* indices,
                   const float* values, uint32_t len, uint32_t* labels,
                   uint32_t label_len) override;

  void Backpropagate(uint32_t batch_indx, const uint32_t* indices,
                     const float* values, float* errors, uint32_t len) {
    BackPropagateImpl<false>(batch_indx, indices, values, errors, len);
  }

  void BackpropagateFirstLayer(uint32_t batch_indx, const uint32_t* indices,
                               const float* values, float* errors,
                               uint32_t len) {
    BackPropagateImpl<true>(batch_indx, indices, values, errors, len);
  }

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
    return active_lens[batch_indx];
  }

  const uint32_t* GetIndices(uint32_t batch_indx) const override {
    return active_neurons[batch_indx];
  }

  const float* GetValues(uint32_t batch_indx) const override {
    return activations[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) override { return errors[batch_indx]; }

  float* GetWeights();

  float* GetBiases();

  ~SparseLayer();

 private:
  template <bool FIRST_LAYER>
  void BackPropagateImpl(uint32_t batch_indx, const uint32_t* indices,
                         const float* values, float* errors, uint32_t len);

  void SelectActiveNeurons(uint32_t batch_indx, const uint32_t* indices,
                           const float* values, uint32_t len, uint32_t* labels,
                           uint32_t label_len);

  constexpr float ActFuncDerivative(float x);

  uint64_t dim, prev_dim, batch_size, sparse_dim;
  float sparsity;
  ActivationFunc act_func;

  uint32_t* active_lens;
  uint32_t** active_neurons;
  float** activations;
  float** errors;

  float* weights;
  float* w_gradient;
  float* w_momentum;
  float* w_velocity;

  float* biases;
  float* b_gradient;
  float* b_momentum;
  float* b_velocity;

  bool* is_active;

  SamplingConfig sampling_config;
  utils::DWTAHashFunction* hasher;
  utils::SampledHashTable<uint32_t>* hash_table;
  uint32_t* rand_neurons;
};

}  // namespace thirdai::bolt
