#pragma once

#include <cereal/access.hpp>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <utils/StringManipulation.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class CountBlock final : public Block {
 public:
  CountBlock(ColumnIdentifier column, char delimiter, uint32_t ceiling)
      : _column(std::move(column)), _delimiter(delimiter), _ceiling(ceiling) {}

  uint32_t featureDim() const final { return _ceiling; }

  bool isDense() const final { return false; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input_row) final {
    (void)index_within_block;
    (void)input_row;
    throw std::invalid_argument(
        "Explanations are not supported by CountBlock.");
  }

  void buildSegment(ColumnarInputSample& input_row,
                    SegmentedFeatureVector& vec) final {
    uint32_t count = text::split(input_row.column(_column), _delimiter).size();
    if (count >= _ceiling) {
      throw std::invalid_argument(
          "Count with ceiling=" + std::to_string(_ceiling) +
          " received a sequence with length=" + std::to_string(count) + ".");
    }
    vec.addSparseFeatureToSegment(/* index= */ count, /* value= */ 1.0);
  }

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_column};
  }

  static auto make(ColumnIdentifier column, char delimiter, uint32_t dim) {
    return std::make_shared<CountBlock>(std::move(column), delimiter, dim);
  }

 private:
  ColumnIdentifier _column;
  char _delimiter;
  uint32_t _ceiling;

  CountBlock() {}

  friend cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::CountBlock)