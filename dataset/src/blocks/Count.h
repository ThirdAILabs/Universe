#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <utils/StringManipulation.h>
#include <limits>
#include <memory>
#include <stdexcept>
namespace thirdai::dataset {

class CountBlock final : public Block {
 public:
  CountBlock(ColumnIdentifier column, char delimiter)
      : _column(std::move(column)), _delimiter(delimiter) {}

  uint32_t featureDim() const final {
    return std::numeric_limits<uint32_t>::max();
  }

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
    uint32_t index = text::split(input_row.column(_column), _delimiter).size();
    vec.addSparseFeatureToSegment(index, /* value= */ 1.0);
  }

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_column};
  }

  static auto make(ColumnIdentifier column, char delimiter) {
    return std::make_shared<CountBlock>(std::move(column), delimiter);
  }

 private:
  ColumnIdentifier _column;
  char _delimiter;
};

}  // namespace thirdai::dataset