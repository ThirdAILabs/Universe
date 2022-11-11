#include "LossFunctions.h"
#include <cereal/types/polymorphic.hpp>

namespace thirdai::bolt {

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

}  // namespace thirdai::bolt
CEREAL_REGISTER_TYPE(thirdai::bolt::CategoricalCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::BinaryCrossEntropyLoss)
CEREAL_REGISTER_TYPE(thirdai::bolt::MeanSquaredError)
CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedMeanAbsolutePercentageErrorLoss)
