#include "ComparativeLoss.h"
#include <_types/_uint32_t.h>
#include <optional>

namespace thirdai::bolt::nn::loss {

ComparativeLoss::ComparativeLoss(tensor::ActivationTensorPtr activations)
    : _activations(std::move(activations)) {
  _labels = tensor::InputTensor::make(_activations->dim(),
                                      tensor::SparsityType::Unknown,
                                      /* num_nonzeros= */ std::nullopt);
}

void ComparativeLoss::gradients(uint32_t index_in_batch, uint32_t batch_size) {
  const BoltVector& labels = _labels->getVector(index_in_batch);
  BoltVector& activations = _activations->getVector(index_in_batch);

  constexpr bool DENSE = true;
  constexpr bool SPARSE = false;

  if (activations.isDense()) {
    if (labels.isDense()) {
      gradients<DENSE, DENSE>(activations, labels, batch_size);
    } else {
      gradients<DENSE, SPARSE>(activations, labels, batch_size);
    }
  } else {
    if (labels.isDense()) {
      gradients<SPARSE, DENSE>(activations, labels, batch_size);

    } else {
      gradients<SPARSE, SPARSE>(activations, labels, batch_size);
    }
  }
}

std::vector<tensor::ActivationTensorPtr> ComparativeLoss::outputsUsed() const {
  return {_activations};
}

template <bool ACT_DENSE, bool LABEL_DENSE>
void ComparativeLoss::gradients(BoltVector& activations,
                                const BoltVector& labels, uint32_t batch_size) {
  assert(ACT_DENSE != activations.active_neurons == nullptr);
  assert(LABEL_DENSE != labels.active_neurons == nullptr);
  if constexpr (ACT_DENSE && LABEL_DENSE) {
    assert(output.len == labels.len);
  }

  /**
   * Loss gradients are only computed during training. If the label is sparse,
   * the neurons of the network's final layer that correspond to the label's
   * nonzero elements are automatically selected and activated during training.
   * Thus, we don't have to consider the case where there are nonzeros in the
   * label that correspond to inactive neurons in the output layer.
   */
  for (uint32_t i = 0; i < activations.len; i++) {
    uint32_t active_neuron = activations.activeNeuronAtIndex<ACT_DENSE>(i);
    float label_val =
        labels.findActiveNeuron<LABEL_DENSE>(active_neuron).activation;
    activations.gradients[i] =
        gradient(activations.activations[i], label_val, batch_size);
  }
}

}  // namespace thirdai::bolt::nn::loss