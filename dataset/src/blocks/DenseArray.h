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
  DenseArrayBlock(uint32_t start_col, uint32_t dim)
      : _start_col(start_col), _dim(dim) {}

  static auto make(uint32_t start_col, uint32_t dim) {
    return std::make_shared<DenseArrayBlock>(start_col, dim);
  }

  static auto makeSingle(uint32_t start_col) {
    return std::make_shared<DenseArrayBlock>(start_col, /* dim= */ 1);
  }

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return true; };

  Explanation explainIndex(uint32_t index_within_block,
                           SingleColumnarInput& input_row) final {
    char* end;
    float value =
        std::strtof(input_row.column(index_within_block).data(), &end);
    return {_start_col.number() + index_within_block, std::to_string(value)};
  }

 protected:
  std::exception_ptr buildSegment(SingleColumnarInput& input_row,
                                  SegmentedFeatureVector& vec) final {
    for (uint32_t i = _start_col.number(); i < _start_col.number() + _dim;
         i++) {
      char* end;
      float value = std::strtof(input_row.column(i).data(), &end);
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

  std::vector<ColumnIdentifier*> getColumnIdentifiers() final {
    return {&_start_col};
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
