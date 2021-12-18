#pragma once

#include <cstdint>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

class Layer {
 public:
  virtual void forward(const uint32_t* indices_in, const float* values_in,
                       uint32_t len_in, uint32_t* indices_out,
                       float* values_out, const uint32_t* labels = nullptr,
                       uint32_t label_len = 0) = 0;

  virtual void backpropagate(const uint32_t* indices_in, const float* values_in,
                             float* gradients_in, uint32_t len_in,
                             const uint32_t* indices_out,
                             const float* values_out,
                             const float* gradients_out) = 0;

  virtual void backpropagateInputLayer(const uint32_t* indices_in,
                                       const float* values_in, uint32_t len_in,
                                       const uint32_t* indices_out,
                                       const float* values_out,
                                       const float* gradients_out) = 0;
};

class LossFunction {
 public:
  virtual void operator()(const uint32_t* indices, const float* values,
                          float* gradients, uint32_t len,
                          const uint32_t* labels, uint32_t label_len) const = 0;
};

}  // namespace thirdai::bolt