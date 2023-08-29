#include "MarginBCE.h"
#include <cereal/archives/binary.hpp>
#include <cereal/details/helpers.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cmath>
#include <optional>

namespace thirdai::bolt {

MarginBCE::MarginBCE(ComputationPtr output, ComputationPtr labels, float margin)
    : ComparativeLoss(std::move(output), std::move(labels)), _positive_margin(margin), _negative_margin(margin) {}

std::shared_ptr<MarginBCE> MarginBCE::make(ComputationPtr output,
                                           ComputationPtr labels, float margin) {
  return std::make_shared<MarginBCE>(std::move(output), std::move(labels), margin);
}

float MarginBCE::singleLoss(float activation, float label) const {
  if (label == 0.0) {
    activation += _negative_margin;
  } else {
    activation -= _positive_margin;
  }
  if (_bound) {
    activation = std::min<float>(activation, 1.0);
    activation = std::max<float>(activation, 0.0);
  }
  return -label * std::log(activation) + (label - 1) * std::log(1 - activation);
}

float MarginBCE::singleGradient(float activation, float label,
                                uint32_t batch_size) const {
  if (label == 0.0) {
    activation += _negative_margin;
  } else {
    activation -= _positive_margin;
  }
  if (_bound) {
    activation = std::min<float>(activation, 1.0);
    activation = std::max<float>(activation, 0.0);
  }
  return (label - activation) / batch_size;
}

template void MarginBCE::serialize(cereal::BinaryInputArchive&);
template void MarginBCE::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MarginBCE::serialize(Archive& archive) {
  archive(cereal::base_class<ComparativeLoss>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::MarginBCE,
                               "thirdai::bolt::nn::loss::BinaryCrossEntropy")