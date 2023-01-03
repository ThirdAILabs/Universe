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

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _column.updateColumnNumber(column_number_map);
  }

  bool hasColumnNames() const final { return _column.hasName(); }

  bool hasColumnNumbers() const final { return _column.hasNumber(); }

  uint32_t featureDim() const override { return 1; };

  bool isDense() const override { return _dense; };

  uint32_t expectedNumColumns() const final { return _column.number() + 1; };

  Explanation explainIndex(uint32_t index_within_block,
                           const RowInput& columnar_sample) final {
    (void)columnar_sample;
    (void)index_within_block;
    throw std::invalid_argument(
        "Explain feature is not yet implemented in mock block!");
  }

  Explanation explainIndex(uint32_t index_within_block,
                           const MapInput& input_map) final {
    (void)input_map;
    (void)index_within_block;
    throw std::invalid_argument(
        "Explain feature is not yet implemented in mock block!");
  }

 protected:
  std::exception_ptr buildSegment(const RowInput& input_row,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_row, vec);
  }

  std::exception_ptr buildSegment(const MapInput& input_map,
                                  SegmentedFeatureVector& vec) final {
    return buildSegmentImpl(input_map, vec);
  }

  template <typename ColumnarInputType>
  std::exception_ptr buildSegmentImpl(const ColumnarInputType& input,
                                      SegmentedFeatureVector& vec) {
    auto val_str = input.at(_column);
    char* end;
    float val = std::strtof(val_str.data(), &end);

    if (_dense) {
      vec.addDenseFeatureToSegment(val);
    } else {
      vec.addSparseFeatureToSegment(0, val);
    }
    return nullptr;
  };

 private:
  ColumnIdentifier _column;
  bool _dense;
};

}  // namespace thirdai::dataset