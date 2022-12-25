#include "LossFunctions.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>

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
  assert(!(OUTPUT_DENSE && output.neurons != nullptr));
  assert(!LABEL_DENSE || labels.neurons == nullptr);
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
    uint32_t neuron = OUTPUT_DENSE ? i : output.neurons[i];
    float label_val = labels.find(neuron).activation;
    output.gradients[i] =
        elementLossGradient(label_val, output.activations[i], batch_size);
  }
}

template <class Archive>
void LossFunction::serialize(Archive& archive) {
  (void)archive;
}

template <class Archive>
void CategoricalCrossEntropyLoss::serialize(Archive& archive) {
  archive(cereal::base_class<LossFunction>(this));
}

template <class Archive>
void BinaryCrossEntropyLoss::serialize(Archive& archive) {
  archive(cereal::base_class<LossFunction>(this));
}

template <class Archive>
void MeanSquaredError::serialize(Archive& archive) {
  archive(cereal::base_class<LossFunction>(this));
}

template <class Archive>
void WeightedMeanAbsolutePercentageErrorLoss::serialize(Archive& archive) {
  archive(cereal::base_class<LossFunction>(this));
}

template <class Archive>
void MarginBCE::serialize(Archive& archive) {
  archive(cereal::base_class<LossFunction>(this), _positive_margin,
          _negative_margin, _bound);
}

template void CategoricalCrossEntropyLoss::serialize(
    cereal::BinaryOutputArchive&);
template void BinaryCrossEntropyLoss::serialize(cereal::BinaryOutputArchive&);
template void MeanSquaredError::serialize(cereal::BinaryOutputArchive&);
template void WeightedMeanAbsolutePercentageErrorLoss::serialize(
    cereal::BinaryOutputArchive&);

template void CategoricalCrossEntropyLoss::serialize(
    cereal::BinaryInputArchive&);
template void BinaryCrossEntropyLoss::serialize(cereal::BinaryInputArchive&);
template void MeanSquaredError::serialize(cereal::BinaryInputArchive&);
template void WeightedMeanAbsolutePercentageErrorLoss::serialize(
    cereal::BinaryInputArchive&);

template void CategoricalCrossEntropyLoss::serialize(
    cereal::PortableBinaryInputArchive&);
template void BinaryCrossEntropyLoss::serialize(
    cereal::PortableBinaryInputArchive&);
template void MeanSquaredError::serialize(cereal::PortableBinaryInputArchive&);
template void WeightedMeanAbsolutePercentageErrorLoss::serialize(
    cereal::PortableBinaryInputArchive&);

template void CategoricalCrossEntropyLoss::serialize(
    cereal::PortableBinaryOutputArchive&);
template void BinaryCrossEntropyLoss::serialize(
    cereal::PortableBinaryOutputArchive&);
template void MeanSquaredError::serialize(cereal::PortableBinaryOutputArchive&);
template void WeightedMeanAbsolutePercentageErrorLoss::serialize(
    cereal::PortableBinaryOutputArchive&);

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::CategoricalCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::BinaryCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MeanSquaredError)
CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedMeanAbsolutePercentageErrorLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MarginBCE)
