#include "BinaryCrossEntropy.h"
#include <algorithm>
#include <cmath>
#include <optional>

namespace thirdai::bolt::nn::loss {

BinaryCrossEntropy::BinaryCrossEntropy(tensor::ActivationTensorPtr activations)
    : ComparativeLoss(std::move(activations)) {}

std::shared_ptr<BinaryCrossEntropy> BinaryCrossEntropy::make(
    tensor::ActivationTensorPtr activations) {
  return std::make_shared<BinaryCrossEntropy>(std::move(activations));
}

float BinaryCrossEntropy::singleLoss(float activation, float label) const {
  activation = std::clamp(activation, 1e-6F, 1 - 1e-6F);

  return -label * std::log(activation) + (label - 1) * std::log(1 - activation);
}

float BinaryCrossEntropy::singleGradient(float activation, float label,
                                         uint32_t batch_size) const {
  return (label - activation) / batch_size;
}

}  // namespace thirdai::bolt::nn::loss