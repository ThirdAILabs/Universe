#include "LossFunctions.h"
#include <algorithm>
#include <cassert>
#include <iterator>

namespace thirdai::bolt {

// template void computeDenseLoss<2>(BoltVector&, const BoltVector&, uint32_t);
// template void computeDenseLoss<1>(BoltVector&, const BoltVector&, uint32_t);

template <uint32_t SCALE, bool OUTPUT_DENSE, bool LABEL_DENSE>
void computeLossImpl(BoltVector& output, const BoltVector& labels,
                     uint32_t batch_size) {
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
        SCALE * (label_val - output.activations[i]) / batch_size;
  }
}

template <uint32_t SCALE>
void computeLoss(BoltVector& output, const BoltVector& labels,
                 uint32_t batch_size) {
  if (output.isDense()) {
    if (labels.isDense()) {
      computeLossImpl<SCALE, true, true>(output, labels, batch_size);
    } else {
      computeLossImpl<SCALE, true, false>(output, labels, batch_size);
    }
  } else {
    if (labels.isDense()) {
      computeLossImpl<SCALE, false, true>(output, labels, batch_size);
    } else {
      computeLossImpl<SCALE, false, false>(output, labels, batch_size);
    }
  }
}

void CategoricalCrossEntropyLoss::operator()(BoltVector& output,
                                             const BoltVector& labels,
                                             uint32_t batch_size) const {
  computeLoss<1>(output, labels, batch_size);
}

void MeanSquaredError::operator()(BoltVector& output, const BoltVector& labels,
                                  uint32_t batch_size) const {
  computeLoss<2>(output, labels, batch_size);
}

}  // namespace thirdai::bolt
