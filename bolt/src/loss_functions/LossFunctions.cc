#include "LossFunctions.h"
#include <algorithm>
#include <cassert>
#include <iterator>

namespace thirdai::bolt {

template void computeDenseLoss<2>(BoltVector&, const BoltVector&, uint32_t);
template void computeDenseLoss<1>(BoltVector&, const BoltVector&, uint32_t);

template <uint32_t SCALE>
void computeDenseLoss(BoltVector& output, const BoltVector& labels,
                      uint32_t batch_size) {
  assert(output.len == labels.len);
  assert(output.isDense() == labels.isDense());

  for (uint32_t i = 0; i < output.len; i++) {
    output.gradients[i] =
        SCALE * (labels.activations[i] - output.activations[i]) / batch_size;
  }
}

template void computeSparseLoss<2>(BoltVector&, const BoltVector&, uint32_t);
template void computeSparseLoss<1>(BoltVector&, const BoltVector&, uint32_t);

template <uint32_t SCALE>
void computeSparseLoss(BoltVector& output, const BoltVector& labels,
                       uint32_t batch_size) {
  assert(output.isDense() == labels.isDense());

  for (uint32_t i = 0; i < output.len; i++) {
    uint32_t active_neuron = output.active_neurons[i];
    const uint32_t* label_start = labels.active_neurons;
    const uint32_t* label_end = labels.active_neurons + labels.len;
    const uint32_t* itr = std::find(label_start, label_end, active_neuron);
    if (itr == label_end) {
      output.gradients[i] = SCALE * (-output.activations[i]) / batch_size;
    } else {
      float correct_val = labels.activations[std::distance(label_start, itr)];
      output.gradients[i] =
          SCALE * (correct_val - output.activations[i]) / batch_size;
    }
  }
}

}  // namespace thirdai::bolt
