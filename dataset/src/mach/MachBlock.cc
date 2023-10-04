#include "MachBlock.h"

namespace thirdai::dataset::mach {

MachBlock::MachBlock(ColumnIdentifier col, MachIndexPtr index,
                     std::optional<char> delimiter, bool normalize_labels)
    : CategoricalBlock(std::move(col),
                       /* dim= */ index->numBuckets(), delimiter),
      _index(std::move(index)),
      _normalize_labels {}

void MachBlock::setIndex(const MachIndexPtr& index) {
  if (_index->numBuckets() != index->numBuckets()) {
    throw std::invalid_argument(
        "Output range mismatch in new index. Index output range should be " +
        std::to_string(_index->numBuckets()) +
        " but provided an index with range = " +
        std::to_string(index->numBuckets()) + ".");
  }

  _index = index;
}

std::string MachBlock::getResponsibleCategory(
    uint32_t index, const std::string_view& category_value) const {
  (void)index;
  (void)category_value;
  throw exceptions::NotImplemented("Explainability not supported.");
}

void MachBlock::encodeCategory(std::string_view category,
                               uint32_t num_categories_in_sample,
                               SegmentedFeatureVector& vec) {
  (void)num_categories_in_sample;

  uint32_t entity = std::strtoul(category.data(), nullptr, 10);

  auto hashes = _index->getHashes(entity);

  if (_normalize_labels) {
    for (const auto& hash : hashes) {
      vec.addSparseFeatureToSegment(hash, 1.0 / hashes.size());
    }
  } else {
    for (const auto& hash : hashes) {
      vec.addSparseFeatureToSegment(hash, 1.0);
    }
  }
}

}  // namespace thirdai::dataset::mach

CEREAL_REGISTER_TYPE(thirdai::dataset::mach::MachBlock)