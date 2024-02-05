#include "ComparativeLoss.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace thirdai::bolt {

ComparativeLoss::ComparativeLoss(ComputationPtr output, ComputationPtr labels)
    : _output(std::move(output)), _labels(std::move(labels)) {
  if (_output->dim() != _labels->dim()) {
    std::stringstream error;
    error << "Cannot have comparative loss between output of dimension "
          << _output->dim() << " and labels of dimension " << _labels->dim()
          << ".";
    throw std::invalid_argument(error.str());
  }
}

float ComparativeLoss::loss(uint32_t index_in_batch) const {
  const BoltVector& labels = _labels->tensor()->getVector(index_in_batch);
  const BoltVector& activations = _output->tensor()->getVector(index_in_batch);

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
  const BoltVector& labels = _labels->tensor()->getVector(index_in_batch);
  BoltVector& activations = _output->tensor()->getVector(index_in_batch);

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

ComputationList ComparativeLoss::outputsUsed() const { return {_output}; }

ComputationList ComparativeLoss::labels() const { return {_labels}; }

template <bool ACT_DENSE, bool LABEL_DENSE>
float ComparativeLoss::loss(const BoltVector& activations,
                            const BoltVector& labels) const {
  float total_loss = 0;
  bolt_vector::visitPair(activations, labels,
                         [&total_loss, this](FoundActiveNeuron act_neuron,
                                             FoundActiveNeuron label_neuron) {
                           total_loss += singleLoss(act_neuron.activation,
                                                    label_neuron.activation);
                         });
  return total_loss;
}

template <bool ACT_DENSE, bool LABEL_DENSE>
void ComparativeLoss::gradients(BoltVector& activations,
                                const BoltVector& labels,
                                uint32_t batch_size) const {
  assert(ACT_DENSE == activations.isDense());
  assert(LABEL_DENSE == labels.isDense());
  if constexpr (ACT_DENSE && LABEL_DENSE) {
    assert(activations.len == labels.len);
  }

  float sum_labels = 0;
  for (uint32_t i = 0; i < labels.len; i++) {
    sum_labels += labels.activations[i];
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
    activations.gradients[i] = singleGradient(
        activations.activations[i], label_val, sum_labels, batch_size);
  }
}

template void ComparativeLoss::serialize(cereal::BinaryInputArchive&);
template void ComparativeLoss::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void ComparativeLoss::serialize(Archive& archive) {
  archive(cereal::base_class<Loss>(this), _output, _labels);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::ComparativeLoss,
                               "thirdai::bolt::nn::loss::ComparativeLoss")
