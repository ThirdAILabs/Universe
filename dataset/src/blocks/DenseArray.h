#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <cmath>
#include <exception>
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
class DenseArrayBlock final : public Block {
 public:
  /**
   * Constructor.
   *
   * Arguments:
   *   start_col: int - the column number of the first dense array entry.
   *   dim: int - the dimension of the dense array; the number of dense values.
   */
  DenseArrayBlock(ColumnIdentifier start_col, uint32_t dim)
      : _start_col(std::move(start_col)), _dim(dim) {}

  static auto make(ColumnIdentifier start_col, uint32_t dim) {
    return std::make_shared<DenseArrayBlock>(std::move(start_col), dim);
  }

  static auto makeSingle(ColumnIdentifier start_col) {
    return std::make_shared<DenseArrayBlock>(std::move(start_col),
                                             /* dim= */ 1);
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _start_col.updateColumnNumber(column_number_map);
  }

  bool hasColumnNames() const final { return _start_col.hasName(); }

  bool hasColumnNumbers() const final { return _start_col.hasNumber(); }

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return _start_col + _dim; };

  Explanation explainIndex(uint32_t index_within_block,
                           const RowInput& input_row) final {
    char* end;
    float value = std::strtof(input_row.at(index_within_block).data(), &end);
    return {_start_col + index_within_block, std::to_string(value)};
  }

  Explanation explainIndex(uint32_t index_within_block,
                           const MapInput& input_map) final {
    (void)index_within_block;
    (void)input_map;
    throw std::invalid_argument(
        "Cannot build DenseArray segment with map input.");
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

  std::exception_ptr buildSegment(const MapInput& input_map,
                                  SegmentedFeatureVector& vec) final {
    (void)input_map;
    (void)vec;
    return std::make_exception_ptr(std::invalid_argument(
        "Cannot build DenseArray segment with map input."));
  }

 private:
  ColumnIdentifier _start_col;
  uint32_t _dim;

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
