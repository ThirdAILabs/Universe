#pragma once

#include <cstdint>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

enum class ActivationFunc { ReLU, Softmax };

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;

  SamplingConfig()
      : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size) {}
};

class Layer {
 public:
  virtual void feedForward(uint32_t batch_indx, const uint32_t* indices,
                           const float* values, uint32_t len, uint32_t* labels,
                           uint32_t label_len) = 0;

  virtual void backpropagate(uint32_t batch_indx, const uint32_t* indices,
                             const float* values, float* errors,
                             uint32_t len) = 0;

  virtual void backpropagateFirstLayer(uint32_t batch_indx,
                                       const uint32_t* indices,
                                       const float* values, float* errors,
                                       uint32_t len) = 0;

  virtual void computeErrors(uint32_t batch_indx, uint32_t batch_size, const uint32_t* labels,
                             uint32_t label_len) = 0;

  virtual void updateParameters(float lr, uint32_t iter, float B1, float B2,
                                float eps) = 0;

  virtual uint32_t getLen(uint32_t batch_indx) const = 0;

  virtual const uint32_t* getIndices(uint32_t batch_indx) const = 0;

  virtual const float* getValues(uint32_t batch_indx) const = 0;

  virtual float* getErrors(uint32_t batch_indx) = 0;

  virtual void buildHashTables() = 0;

  virtual void reBuildHashFunction() = 0;

  virtual void setSparsity(float new_sparsity) = 0;

  virtual void setBatchSize(uint64_t new_batch_size) = 0;

  virtual void shuffleRandNeurons() = 0;

  virtual ~Layer() {}
};

}  // namespace thirdai::bolt
