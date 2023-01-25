#include "CategoricalCrossEntropy.h"
#include <cmath>
#include <optional>

namespace thirdai::bolt::nn::loss {

CategoricalCrossEntropy::CategoricalCrossEntropy(
    autograd::ComputationPtr output, autograd::ComputationPtr labels)
    : ComparativeLoss(std::move(output), std::move(labels)) {}

std::shared_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::make(
    autograd::ComputationPtr output, autograd::ComputationPtr labels) {
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

}  // namespace thirdai::bolt::nn::loss