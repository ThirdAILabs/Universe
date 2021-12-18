#include "Layer.h"

namespace thirdai::bolt {

class CrossEntropyLoss final : public LossFunction {
 private:
  template <bool DENSE>
  void computeLoss(const uint32_t* indices, const float* values,
                   float* gradients, uint32_t len, const uint32_t* labels,
                   uint32_t label_len) const {
    float frac = 1.0 / label_len;

    for (uint64_t n = 0; n < len; n++) {
      // Because DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t act_neuron = DENSE ? n : indices[n];
      if (std::find(labels, labels + label_len, act_neuron) !=
          labels + label_len) {
        gradients[n] = (frac - values[n]) / batch_size;
      } else {
        gradients[n] = -values[n] / batch_size;
      }
    }
  }

 public:
  void operator()(const uint32_t* indices, const float* values,
                  float* gradients, uint32_t len, const uint32_t* labels,
                  uint32_t label_len) const override {
    if (indices != nullptr) {
      computeLoss<true>(indices, values, gradients, len, labels, label_len);

    } else {
      computeLoss<false>(indices, values, gradients, len, labels, label_len);
    }
  }
};

class MeanSquaredError final : public LossFunction {
 private:
  template <bool DENSE>
  void computeLoss(const uint32_t* indices, const float* values,
                   float* gradients, uint32_t len, const uint32_t* labels,
                   uint32_t label_len) const {
                     
                   }

 public:
  void operator()(const uint32_t* indices, const float* values,
                  float* gradients, uint32_t len, const uint32_t* labels,
                  uint32_t label_len) const override {
    if (indices != nullptr) {
      computeLoss<true>(indices, values, gradients, len, labels, label_len);

    } else {
      computeLoss<false>(indices, values, gradients, len, labels, label_len);
    }
  }
};

}  // namespace thirdai::bolt
