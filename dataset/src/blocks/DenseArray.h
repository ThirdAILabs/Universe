#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

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
   *   ranges: vector<pair<double, double>> - min
   */
  DenseArrayBlock(uint32_t start_col, uint32_t dim,
                  std::optional<std::vector<std::pair<double, double>>> ranges =
                      std::nullopt)
      : _start_col(start_col), _dim(dim), _ranges(std::move(ranges)) {}

  static auto make(uint32_t start_col, uint32_t dim) {
    return std::make_shared<DenseArrayBlock>(start_col, dim);
  }

  static auto makeSingle(uint32_t start_col) {
    return std::make_shared<DenseArrayBlock>(start_col, /* dim= */ 1);
  }

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return _start_col + _dim; };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    char* end;
    float value = std::strtof(input_row.at(index_within_block).data(), &end);
    return {_start_col + index_within_block, std::to_string(value)};
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    for (uint32_t i = _start_col; i < _start_col + _dim; i++) {
      char* end;
      float value = std::strtof(input_row.at(i).data(), &end);
      if (std::isinf(value)) {
        value = 0;
      }
      if (std::isnan(value)) {
        value = 0;
      }
      vec.addDenseFeatureToSegment(value);
    }
    return nullptr;
  }

 private:
  uint32_t _start_col;
  uint32_t _dim;
  std::optional<std::vector<std::pair<double, double>>> _ranges;

  // Private constructor for cereal.
  DenseArrayBlock() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _start_col, _dim);
  }
};

using DenseArrayBlockPtr = std::shared_ptr<DenseArrayBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::DenseArrayBlock)
