#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/ContiguousNumericId.h>
#include <cmath>
#include <memory>

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

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
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