#include "ComparativeLoss.h"
#include <optional>

namespace thirdai::bolt::nn::loss {

ComparativeLoss::ComparativeLoss(tensor::ActivationTensorPtr activations)
    : _activations(std::move(activations)) {
  _labels = tensor::InputTensor::make(_activations->dim(),
                                      tensor::SparsityType::Unknown,
                                      /* num_nonzeros= */ std::nullopt);
}

float ComparativeLoss::loss(uint32_t index_in_batch) const {
  const BoltVector& labels = _labels->getVector(index_in_batch);
  const BoltVector& activations = _activations->getVector(index_in_batch);

  constexpr bool DENSE = true;
  constexpr bool SPARSE = false;

  if (activations.isDense()) {
    if (labels.isDense()) {
      return loss<DENSE, DENSE>(activations, labels);
    }
    return loss<DENSE, SPARSE>(activations, labels);
  }
  if (labels.isDense()) {
    return loss<SPARSE, DENSE>(activations, labels);
  }
  return loss<SPARSE, SPARSE>(activations, labels);
}

void ComparativeLoss::gradients(uint32_t index_in_batch,
                                uint32_t batch_size) const {
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
float ComparativeLoss::loss(const BoltVector& activations,
                            const BoltVector& labels) const {
  assert(ACT_DENSE != activations.active_neurons == nullptr);
  assert(LABEL_DENSE != labels.active_neurons == nullptr);
  if constexpr (ACT_DENSE && LABEL_DENSE) {
    assert(output.len == labels.len);
  }

  if constexpr (ACT_DENSE || LABEL_DENSE) {
    float total_loss = 0.0;
    uint32_t dim = std::max(activations.len, labels.len);
    for (uint32_t i = 0; i < dim; i++) {
      float activation = activations.findActiveNeuron<ACT_DENSE>(i).activation;
      float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
      total_loss += loss(activation, label);
    }
    return total_loss;
  }

  /**
   * If both are sparse then we need to iterate over the nonzeros from both
   * vectors. To avoid double counting the overlapping neurons we only compute
   * the loss for overlapping neurons when iterating over the activations.
   */
  float total_loss = 0.0;
  for (uint32_t i = 0; i < activations.len; i++) {
    float label =
        labels.findActiveNeuron<LABEL_DENSE>(activations.active_neurons[i])
            .activation;
    float activation = activations.activations[i];
    total_loss += loss(activation, label);
  }

  for (uint32_t i = 0; i < labels.len; i++) {
    auto activation_neuron =
        activations.findActiveNeuron<ACT_DENSE>(labels.active_neurons[i]);
    // Skip any neurons that were in the activations since the loss was already
    // computed for them.
    if (!activation_neuron.pos) {
      float label = labels.activations[i];
      // The activation is 0 since this isn't in the output active neurons.
      total_loss += loss(/* activation= */ 0.0, label);
    }
  }
  return total_loss;
}

template <bool ACT_DENSE, bool LABEL_DENSE>
void ComparativeLoss::gradients(BoltVector& activations,
                                const BoltVector& labels,
                                uint32_t batch_size) const {
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