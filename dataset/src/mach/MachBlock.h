#pragma once

#include "MachIndex.h"
#include <dataset/src/blocks/Categorical.h>
#include <exceptions/src/Exceptions.h>
#include <variant>

namespace thirdai::dataset::tests {
class MachBlockTest;
}  // namespace thirdai::dataset::tests

namespace thirdai::dataset::mach {

/**
 * A MachBlock applies to a a single column with potentially more than one
 * category (as specified by the delimiter). For each category found, we will
 * update the given MachIndex by calls to hashEntity. This will hash the
 * given category some number of times and return those hashes. It will also
 * store an inverted index from hashes to entities.
 */
class MachBlock final : public CategoricalBlock {
 public:
  MachBlock(ColumnIdentifier col, MachIndexPtr index,
            std::optional<char> delimiter = std::nullopt,
            bool normalize_labels = false);

  static auto make(ColumnIdentifier col, MachIndexPtr index,
                   std::optional<char> delimiter = std::nullopt,
                   bool normalize_labels = false) {
    return std::make_shared<MachBlock>(std::move(col), std::move(index),
                                       delimiter, normalize_labels);
  }

  MachIndexPtr index() const { return _index; }

  void setIndex(const MachIndexPtr& index);

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final;

  friend class tests::MachBlockTest;

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final;

 private:
  MachIndexPtr _index;
  bool _normalize_labels;

  // Private constructor for cereal.
  MachBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _index);
  }
};

using MachBlockPtr = std::shared_ptr<MachBlock>;

}  // namespace thirdai::dataset::mach