#include "BinaryCrossEntropy.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <archive/src/Map.h>
#include <algorithm>
#include <cmath>
#include <optional>

namespace thirdai::bolt {

BinaryCrossEntropy::BinaryCrossEntropy(ComputationPtr output,
                                       ComputationPtr labels)
    : ComparativeLoss(std::move(output), std::move(labels)) {}

std::shared_ptr<BinaryCrossEntropy> BinaryCrossEntropy::make(
    ComputationPtr output, ComputationPtr labels) {
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

ar::ConstArchivePtr BinaryCrossEntropy::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));
  map->set("output", ar::str(_output->name()));
  map->set("labels", ar::str(_labels->name()));

  return map;
}

std::shared_ptr<BinaryCrossEntropy> BinaryCrossEntropy::fromArchive(
    const ar::Archive& archive,
    const std::unordered_map<std::string, ComputationPtr>& computations) {
  assertLossType(archive, type());

  return BinaryCrossEntropy::make(computations.at(archive.str("output")),
                                  computations.at(archive.str("labels")));
}

template void BinaryCrossEntropy::serialize(cereal::BinaryInputArchive&);
template void BinaryCrossEntropy::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void BinaryCrossEntropy::serialize(Archive& archive) {
  archive(cereal::base_class<ComparativeLoss>(this));
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::BinaryCrossEntropy,
                               "thirdai::bolt::nn::loss::BinaryCrossEntropy")