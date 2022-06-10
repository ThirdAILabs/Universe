#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <charconv>
#include <string_view>

namespace thirdai::dataset {

/**
 * A mock block that parses a floating point number
 * in the specified column and produces a one-dimensional
 * vector with the number as its value.
 */
class MockBlock : public Block {
 public:
  explicit MockBlock(uint32_t column, bool dense)
      : _column(column), _dense(dense) {}

  uint32_t featureDim() const override { return 1; };

  bool isDense() const override { return _dense; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) override {
    auto val_str = input_row.at(_column);
    float val{};
    std::from_chars(val_str.data(), val_str.data() + val_str.size(), val, std::chars_format::general);
    
    if (_dense) {
      vec.addDenseFeatureToSegment(val);
    } else {
      vec.addSparseFeatureToSegment(0, val);
    }
  };

 private:
  uint32_t _column;
  bool _dense;
};

}  // namespace thirdai::dataset