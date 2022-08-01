#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <cstdlib>

namespace thirdai::dataset {

/**
 * A block that parses a floating point number
 * in the specified column and produces a one-dimensional
 * dense vector with the number as its value.
 */
class ContinuousBlock : public Block {
 public:
  explicit ContinuousBlock(uint32_t column) : _column(column) {}

  uint32_t featureDim() const override { return 1; };

  bool isDense() const override { return true; };

  uint32_t expectedNumColumns() const final { return _column + 1; };

  std::string giveMessage(
      float gradient_ratio_value,
      std::unordered_map<uint32_t, std::string> col_num_col_name_map,
      float row_ratio_sum, bool to_print_message) const final {
    if (to_print_message) {
      std::cout << "The Continuous column  " +
                       col_num_col_name_map.at(_column) + " is  " +
                       std::to_string(
                           ((gradient_ratio_value) / (row_ratio_sum)) * 100) +
                       "% responsible."
                << std::endl;
    }
    return col_num_col_name_map.at(_column);
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) override {
    auto val_str = input_row.at(_column);
    char* end;
    float val = std::strtof(val_str.data(), &end);

    vec.addDenseFeatureToSegment(val);

    return nullptr;
  };

 private:
  uint32_t _column;
};

}  // namespace thirdai::dataset