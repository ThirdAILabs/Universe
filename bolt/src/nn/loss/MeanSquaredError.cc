#include "MeanSquaredError.h"
#include <cereal/archives/binary.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cmath>

namespace thirdai::bolt::nn::loss {

MeanSquaredError::MeanSquaredError(autograd::ComputationPtr output,
                                   autograd::ComputationPtr labels)
    : ComparativeLoss(std::move(output), std::move(labels)) {}

std::shared_ptr<MeanSquaredError> MeanSquaredError::make(
    autograd::ComputationPtr output, autograd::ComputationPtr labels) {
  return std::make_shared<MeanSquaredError>(std::move(output),
                                            std::move(labels));
}

float MeanSquaredError::singleLoss(float activation, float label) const {
  return -std::pow(label - activation, 2.0);
}

float MeanSquaredError::singleGradient(float activation, float label,
                                       uint32_t batch_size) const {
  return 2 * (label - activation) / batch_size;
}

template void MeanSquaredError::serialize(cereal::BinaryInputArchive&);
template void MeanSquaredError::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MeanSquaredError::serialize(Archive& archive) {
  archive(cereal::base_class<ComparativeLoss>(this));
}

}  // namespace thirdai::bolt::nn::loss

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::loss::MeanSquaredError)