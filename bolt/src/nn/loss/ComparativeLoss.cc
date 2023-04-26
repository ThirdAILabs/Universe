#include "ComparativeLoss.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <bolt_vector/src/BoltVectorUtils.h>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace thirdai::bolt::nn::loss {

ComparativeLoss::ComparativeLoss(autograd::ComputationPtr output,
                                 autograd::ComputationPtr labels)
    : _output(std::move(output)), _labels(std::move(labels)) {
  if (_output->dims() != _labels->dims()) {
    throw std::invalid_argument(
        "Cannot compute a comparative loss between output of dimension " +
        tensor::toString(_output->dims()) + " and labels of dimension " +
        tensor::toString(_labels->dims()) + ".");
  }
}

namespace {
constexpr bool DENSE = true;
constexpr bool SPARSE = false;
}  // namespace

float ComparativeLoss::loss(uint32_t index_in_batch) const {
  const auto& activations = _output->tensor();
  const auto& labels = _labels->tensor();

  uint32_t start = activations->rangeStart(index_in_batch);
  uint32_t end = activations->rangeEnd(index_in_batch);

  float total_loss = 0;
  for (uint32_t i = start; i < end; i++) {
    const BoltVector& act_vec = activations->getVector(i);
    const BoltVector& label_vec = labels->getVector(i);
    if (act_vec.isDense()) {
      if (label_vec.isDense()) {
        total_loss += loss<DENSE, DENSE>(act_vec, label_vec);
      } else {
        total_loss += loss<DENSE, SPARSE>(act_vec, label_vec);
      }
    } else {
      if (label_vec.isDense()) {
        total_loss += loss<SPARSE, DENSE>(act_vec, label_vec);
      } else {
        total_loss += loss<SPARSE, SPARSE>(act_vec, label_vec);
      }
    }
  }

  return total_loss / (end - start);
}

void ComparativeLoss::gradients(uint32_t index_in_batch,
                                uint32_t batch_size) const {
  auto& activations = _output->tensor();
  const auto& labels = _labels->tensor();

  uint32_t start = activations->rangeStart(index_in_batch);
  uint32_t end = activations->rangeEnd(index_in_batch);

  for (uint32_t i = start; i < end; i++) {
    BoltVector& act_vec = activations->getVector(i);
    const BoltVector& label_vec = labels->getVector(i);
    if (act_vec.isDense()) {
      if (label_vec.isDense()) {
        gradients<DENSE, DENSE>(act_vec, label_vec, batch_size);
      } else {
        gradients<DENSE, SPARSE>(act_vec, label_vec, batch_size);
      }
    } else {
      if (label_vec.isDense()) {
        gradients<SPARSE, DENSE>(act_vec, label_vec, batch_size);
      } else {
        gradients<SPARSE, SPARSE>(act_vec, label_vec, batch_size);
      }
    }
  }
}

autograd::ComputationList ComparativeLoss::outputsUsed() const {
  return {_output};
}

autograd::ComputationList ComparativeLoss::labels() const { return {_labels}; }

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
        singleGradient(activations.activations[i], label_val, batch_size);
  }
}

template void ComparativeLoss::serialize(cereal::BinaryInputArchive&);
template void ComparativeLoss::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void ComparativeLoss::serialize(Archive& archive) {
  archive(cereal::base_class<Loss>(this), _output, _labels);
}

}  // namespace thirdai::bolt::nn::loss

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::loss::ComparativeLoss)
