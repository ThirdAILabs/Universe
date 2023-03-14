#pragma once

#include "Categorical.h"
#include "MachIndex.h"
#include <exceptions/src/Exceptions.h>
#include <variant>

namespace thirdai::dataset {

namespace tests {
class MachBlockTest;
}  // namespace tests

class MachBlock final : public CategoricalBlock {
 public:
  MachBlock(ColumnIdentifier col, MachIndexPtr index,
            std::optional<char> delimiter = std::nullopt);

  static auto make(ColumnIdentifier col, MachIndexPtr index,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<MachBlock>(std::move(col), std::move(index),
                                       delimiter);
  }

  MachIndexPtr index() const { return _index; }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final;

  friend class tests::MachBlockTest;

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final;

 private:
  MachIndexPtr _index;

  // Private constructor for cereal.
  MachBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _index);
  }
};

using MachBlockPtr = std::shared_ptr<MachBlock>;

}  // namespace thirdai::dataset