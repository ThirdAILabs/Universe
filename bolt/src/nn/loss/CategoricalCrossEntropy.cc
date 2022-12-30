#include "CategoricalCrossEntropy.h"
#include <optional>

namespace thirdai::bolt::nn::loss {

CategoricalCrossEntropy::CategoricalCrossEntropy(
    tensor::ActivationTensorPtr activations)
    : ComparativeLoss(std::move(activations)) {}

std::shared_ptr<CategoricalCrossEntropy> CategoricalCrossEntropy::make(
    tensor::ActivationTensorPtr activations) {
  return std::make_shared<CategoricalCrossEntropy>(std::move(activations));
}

float CategoricalCrossEntropy::gradient(float activation, float label,
                                        uint32_t batch_size) {
  return (label - activation) / batch_size;
}

}  // namespace thirdai::bolt::nn::loss