#include "BinaryCrossEntropy.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <algorithm>
#include <cmath>
#include <optional>

namespace thirdai::bolt::nn::loss {

BinaryCrossEntropy::BinaryCrossEntropy(autograd::ComputationPtr output,
                                       autograd::ComputationPtr labels)
    : ComparativeLoss(std::move(output), std::move(labels)) {}

std::shared_ptr<BinaryCrossEntropy> BinaryCrossEntropy::make(
    autograd::ComputationPtr output, autograd::ComputationPtr labels) {
  return std::make_shared<BinaryCrossEntropy>(std::move(output),
                                              std::move(labels));
}

float BinaryCrossEntropy::singleLoss(float activation, float label) const {
  activation = std::clamp(activation, 1e-6F, 1 - 1e-6F);

  return -label * std::log(activation) + (label - 1) * std::log(1 - activation);
}

float BinaryCrossEntropy::singleGradient(float activation, float label,
                                         uint32_t batch_size) const {
  return (label - activation) / batch_size;
}

template <class Archive>
void BinaryCrossEntropy::serialize(Archive& archive) {
  archive(cereal::base_class<ComparativeLoss>(this));
}

}  // namespace thirdai::bolt::nn::loss

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::loss::BinaryCrossEntropy)