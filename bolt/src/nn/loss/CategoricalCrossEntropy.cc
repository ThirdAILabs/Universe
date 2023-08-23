#include "CategoricalCrossEntropy.h"
#include <cereal/archives/binary.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cmath>
#include <optional>

namespace thirdai::bolt {

CategoricalCrossEntropy::CategoricalCrossEntropy(ComputationPtr output,
                                                 ComputationPtr labels)
    : ComparativeLoss(std::move(output), std::move(labels)) {}

std::shared_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::make(
    ComputationPtr output, ComputationPtr labels) {
  return std::make_shared<CategoricalCrossEntropy>(std::move(output),
                                                   std::move(labels));
}

float CategoricalCrossEntropy::singleLoss(float activation, float label) const {
  // We add an small epsilon here to avoid log(0) in the case where the output
  // is sparse and there is a nonzero label that is not among the active
  // neurons.
  if (label == 0) {
    return 0.0;
  }

  // Ensures the activation cannot be zero for numerical stability, also handles
  // the case where the activation is 0 due to sparsity.
  activation = std::max(activation, 1e-6F);

  return -label * std::log(activation);
}

float CategoricalCrossEntropy::singleGradient(float activation, float label,
                                              uint32_t batch_size) const {
  return (label - activation) / batch_size;
}

template void CategoricalCrossEntropy::serialize(cereal::BinaryInputArchive&);
template void CategoricalCrossEntropy::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void CategoricalCrossEntropy::serialize(Archive& archive) {
  archive(cereal::base_class<ComparativeLoss>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(
    thirdai::bolt::CategoricalCrossEntropy,
    "thirdai::bolt::nn::loss::CategoricalCrossEntropy")