#pragma once

#include "DWTA.h"
#include "HashTable.h"
#include <stdint.h>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

enum class ActivationFunc { ReLU, Softmax };

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;

  SamplingConfig() : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables, uint32_t range_pow,
                 uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table), num_tables(num_tables), range_pow(range_pow), reservoir_size(reservoir_size) {}
};

class Layer {
 public:
  Layer() {}

  Layer(const Layer&) = delete;
  Layer(Layer&&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer& operator=(Layer&&) = delete;

  Layer(uint64_t _dim, uint64_t _prev_dim, float _sparsity,
        ActivationFunc _act_func, SamplingConfig _sampling_config);

  void ForwardPass(uint32_t batch_indx, const uint32_t* indices,
                   const float* values, uint32_t len,
                   uint32_t* labels = nullptr, uint32_t label_len = 0);

  template <bool FIRST_LAYER>
  void BackPropagate(uint32_t batch_indx, const uint32_t* indices,
                     const float* values, float* errors, uint32_t len);

  void ComputeErrors(uint32_t batch_indx, const uint32_t* labels,
                     uint32_t label_len);

  void UpdateParameters(float lr, uint32_t iter, float B1 = BETA1,
                        float B2 = BETA2, float eps = EPS);

  void BuildHashTables();

  void ReBuildHashFunction();

  void SetSparsity(float new_sparsity);

  void SetBatchSize(uint64_t new_batch_size);

  void ShuffleRandNeurons();

  uint32_t GetLen(uint32_t batch_indx) { return active_lens[batch_indx]; }

  const uint32_t* GetIndices(uint32_t batch_indx) {
    return active_neurons[batch_indx];
  }

  const float* GetValues(uint32_t batch_indx) {
    return activations[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) { return errors[batch_indx]; }

  float* GetWeights();

  float* GetBiases();

  ~Layer();

 private:
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
  DWTAHashFunction* hasher;
  HashTable<uint32_t, uint32_t>* hash_table;
  uint32_t* rand_neurons;
};

}  // namespace thirdai::bolt
