#pragma once

#include <bolt/layers/BoltVector.h>

namespace thirdai::bolt {

class SparseCategoricalCrossEntropyLoss {
 private:
  template <bool DENSE>
  void computeLoss(BoltVector& output, uint32_t batch_size,
                   const uint32_t* labels, uint32_t label_len) const {
    float frac = 1.0 / label_len;

    for (uint64_t n = 0; n < output.len; n++) {
      // Because DENSE is known at compile time the compiler can remove
      // this conditional
      uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
      if (std::find(labels, labels + label_len, act_neuron) !=
          labels + label_len) {
        output.gradients[n] = (frac - output.activations[n]) / batch_size;
      } else {
        output.gradients[n] = -output.activations[n] / batch_size;
      }
    }
  }

 public:
  void operator()(BoltVector& output, uint32_t batch_size,
                  const uint32_t* labels, uint32_t label_len) const {
    if (output.active_neurons == nullptr) {
      computeLoss<true>(output, batch_size, labels, label_len);

    } else {
      computeLoss<false>(output, batch_size, labels, label_len);
    }
  }
};

class MeanSquaredError {
 private:
  template <bool DENSE, bool TRUTH_DENSE>
  void computeLoss(BoltVector& output, uint32_t batch_size,
                   const uint32_t* truth_indices, const float* truth_values,
                   uint32_t truth_len) const {
    for (uint64_t n = 0; n < output.len; n++) {
      uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
      float matching_truth_value;
      if (TRUTH_DENSE) {
        matching_truth_value = truth_values[act_neuron];
      } else {
        const unsigned int* itr =
            std::find(truth_indices, truth_indices + truth_len, act_neuron);
        if (itr != truth_indices + truth_len) {
          matching_truth_value =
              truth_values[std::distance(truth_indices, itr)];
        } else {
          matching_truth_value = 0.0;
        }
      }
      output.gradients[n] =
          2 * (matching_truth_value - output.activations[n]) / batch_size;
    }
  }

 public:
  void operator()(BoltVector& output, uint32_t batch_size,
                  const uint32_t* truth_indices, const float* truth_values,
                  uint32_t truth_len) const {
    if (output.active_neurons == nullptr) {
      if (truth_indices == nullptr) {
        computeLoss<true, true>(output, batch_size, truth_indices, truth_values,
                                truth_len);

      } else {
        computeLoss<true, false>(output, batch_size, truth_indices,
                                 truth_values, truth_len);
      }
    } else {
      if (truth_indices == nullptr) {
        computeLoss<false, true>(output, batch_size, truth_indices,
                                 truth_values, truth_len);

      } else {
        computeLoss<false, false>(output, batch_size, truth_indices,
                                  truth_values, truth_len);
      }
    }
  }
};

}  // namespace thirdai::bolt
