#pragma once

#include <bolt/src/layers/BoltVector.h>

namespace thirdai::bolt {

class LossFunction {
 public:
  LossFunction() {}

  void operator()(BoltVector& output, const BoltVector& labels,
                  uint32_t batch_size) const {
    if (output.isDense()) {
      if (labels.isDense()) {
        computeLossImpl<true, true>(output, labels, batch_size);
      } else {
        computeLossImpl<true, false>(output, labels, batch_size);
      }
    } else {
      if (labels.isDense()) {
        computeLossImpl<false, true>(output, labels, batch_size);
      } else {
        computeLossImpl<false, false>(output, labels, batch_size);
      }
    }
  }

  virtual ~LossFunction() {}

 private:
  template <bool OUTPUT_DENSE, bool LABEL_DENSE>
  void computeLossImpl(BoltVector& output, const BoltVector& labels,
                       uint32_t batch_size) const {
    assert(!OUTPUT_DENSE || output.active_neurons == nullptr);
    assert(!LABEL_DENSE || labels.active_neurons == nullptr);
    if (OUTPUT_DENSE && LABEL_DENSE) {
      assert(output.len == labels.len);
    }

    for (uint32_t i = 0; i < output.len; i++) {
      uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
      float label_val;
      if (LABEL_DENSE) {
        label_val = labels.activations[active_neuron];
      } else {
        const uint32_t* label_start = labels.active_neurons;
        const uint32_t* label_end = labels.active_neurons + labels.len;
        const uint32_t* itr = std::find(label_start, label_end, active_neuron);
        if (itr == label_end) {
          label_val = 0.0;
        } else {
          label_val = labels.activations[std::distance(label_start, itr)];
        }
      }
      output.gradients[i] =
          elementLoss(label_val, output.activations[i], batch_size);
    }
  }

  virtual float elementLoss(float label, float activation,
                            uint32_t batch_size) const = 0;
};

class CategoricalCrossEntropyLoss final : public LossFunction {
 private:
  float elementLoss(float label, float activation,
                    uint32_t batch_size) const override {
    return (label - activation) / batch_size;
  }
};

class MeanSquaredError final : public LossFunction {
 private:
  float elementLoss(float label, float activation,
                    uint32_t batch_size) const override {
    return 2 * (label - activation) / batch_size;
  }
};

}  // namespace thirdai::bolt
