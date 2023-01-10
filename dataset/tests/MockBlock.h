#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <string_view>

namespace thirdai::dataset {

/**
 * A mock block that parses a floating point number
 * in the specified column and produces a one-dimensional
 * vector with the number as its value.
 */
class MockBlock final : public Block {
 public:
  explicit MockBlock(ColumnIdentifier column, bool dense)
      : _column(std::move(column)), _dense(dense) {}

  uint32_t featureDim() const override { return 1; }

  bool isDense() const override { return _dense; }

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& columnar_sample) final {
    (void)columnar_sample;
    (void)index_within_block;
    throw std::invalid_argument(
        "Explain feature is not yet implemented in mock block!");
  }

  std::vector<ColumnIdentifier*> getColumnIdentifiers() final {
    return {&_column};
  }

 protected:
  std::exception_ptr buildSegment(ColumnarInputSample& input,
                                  SegmentedFeatureVector& vec) final {
    auto val_str = input.column(_column);
    char* end;
    float val = std::strtof(val_str.data(), &end);

    if (_dense) {
      vec.addDenseFeatureToSegment(val);
    } else {
      vec.addSparseFeatureToSegment(0, val);
    }
    return nullptr;
  }

 private:
  ColumnIdentifier _column;
  bool _dense;
};

}  // namespace thirdai::dataset