#include "CategoricalCrossEntropy.h"
#include <cmath>
#include <optional>

namespace thirdai::bolt::nn::loss {

CategoricalCrossEntropy::CategoricalCrossEntropy(
    tensor::ActivationTensorPtr activations)
    : ComparativeLoss(std::move(activations)) {}

std::shared_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::make(
    tensor::ActivationTensorPtr activations) {
  return std::make_shared<CategoricalCrossEntropy>(std::move(activations));
}

float CategoricalCrossEntropy::singleLoss(float activation, float label) const {
  // We add an small epsilon here to avoid log(0) in the case where the output
  // is sparse and there is a nonzero label that is not among the active
  // neurons.
  if (label == 0) {
    return 0.0;
  }
  return -label * std::log(activation + 1e-7);
}

float CategoricalCrossEntropy::singleGradient(float activation, float label,
                                              uint32_t batch_size) const {
  return (label - activation) / batch_size;
}

}  // namespace thirdai::bolt::nn::loss