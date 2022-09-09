#pragma once

#include <dataset/src/blocks/BlockInterface.h>
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

  uint32_t expectedNumColumns() const final { return _column + 1; };

  uint32_t getColumnNum() const final { return _column; }

  std::pair<std::string, std::string> explainIndex(
      uint32_t index,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name)
      const final {
    (void)index;
    (void)num_to_name;
    throw std::invalid_argument("not yet implemented in mock block!");
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec, bool store_map) override {
    (void)store_map;
    auto val_str = input_row.at(_column);
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
  uint32_t _column;
  bool _dense;
};

}  // namespace thirdai::dataset