#include "LossFunctions.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>

namespace thirdai::bolt {

void LossFunction::lossGradients(BoltVector& output, const BoltVector& labels,
                                 uint32_t batch_size) const {
  if (output.isDense()) {
    if (labels.isDense()) {
      computeLossGradientsImpl<true, true>(output, labels, batch_size);
    } else {
      computeLossGradientsImpl<true, false>(output, labels, batch_size);
    }
  } else {
    if (labels.isDense()) {
      computeLossGradientsImpl<false, true>(output, labels, batch_size);
    } else {
      computeLossGradientsImpl<false, false>(output, labels, batch_size);
    }
  }
}

template <bool OUTPUT_DENSE, bool LABEL_DENSE>
void LossFunction::computeLossGradientsImpl(BoltVector& output,
                                            const BoltVector& labels,
                                            uint32_t batch_size) const {
  assert(!(OUTPUT_DENSE && output.active_neurons != nullptr));
  assert(!LABEL_DENSE || labels.active_neurons == nullptr);
  if (OUTPUT_DENSE && LABEL_DENSE) {
    assert(output.len == labels.len);
  }
  /*
    Loss functions are only used in training.
    If the label is sparse, the neurons of the network's final
    layer that correspond to the label's nonzero elements are
    automatically selected and activated during training.
    Thus, we don't have to consider the case where there are
    nonzeros in the label that correspond to inactive neurons in
    the output layer.
  */
  for (uint32_t i = 0; i < output.len; i++) {
    uint32_t active_neuron = OUTPUT_DENSE ? i : output.active_neurons[i];
    float label_val =
        labels.findActiveNeuron<LABEL_DENSE>(active_neuron).activation;
    output.gradients[i] =
        elementLossGradient(label_val, output.activations[i], batch_size);
  }
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::CategoricalCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::BinaryCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MeanSquaredError)
CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedMeanAbsolutePercentageErrorLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MarginBCE)