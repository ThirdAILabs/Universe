#pragma once

#include "BlockInterface.h"
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * A block that adds a dense array segment.
 * Empty columns or undefined elements of the array
 * default to 0. NaNs and Infs also default to 0.
 */
class DenseArrayBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * Arguments:
   *   start_col: int - the column number of the first dense array entry.
   *   dim: int - the dimension of the dense array; the number of dense values.
   */
  DenseArrayBlock(uint32_t start_col, uint32_t dim)
      : _start_col(start_col), _dim(dim) {}

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return _start_col + _dim; };

  ResponsibleColumnAndInputKey explainFeature(
      uint32_t index_within_block,
      std::optional<std::unordered_map<uint32_t, std::string>> num_to_name)
      const final {
    (void)index_within_block;
    (void)num_to_name;
    throw std::invalid_argument("not yet implemented in dense array block!");
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec, bool remember_raw_features) final {
    (void)remember_raw_features;
    for (uint32_t i = _start_col; i < _start_col + _dim; i++) {
      char* end;
      float value = std::strtof(input_row.at(i).data(), &end);
      if (std::isinf(value)) {
        value = 0;
        std::cout << "[DenseArrayBlock] WARNING: Found inf. Defaulting to 0."
                  << std::endl;
      }
      if (std::isnan(value)) {
        value = 0;
        std::cout << "[DenseArrayBlock] WARNING: Found NaN. Defaulting to 0."
                  << std::endl;
      }
      vec.addDenseFeatureToSegment(value);
    }
    return nullptr;
  }

 private:
  uint32_t _start_col;
  uint32_t _dim;
};

}  // namespace thirdai::dataset