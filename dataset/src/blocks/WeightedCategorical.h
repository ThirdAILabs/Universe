#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/weighted_categorical/WeightedCategoricalEncodingInterface.h>
#include <dataset/src/encodings/weighted_categorical/WeightedContiguousNumericId.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an
 * identification string).
 */
class WeightedCategoricalBlock : public Block {
 public:
  // Declaration included from BlockInterface.h
  friend CategoricalBlockTest;

  /**
   * Constructor.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   encoding: CategoricalEncoding - the categorical feature encoding model.
   */
  WeightedCategoricalBlock(uint32_t col, std::shared_ptr<WeightedCategoricalEncoding> encoding)
      : _col(col), _encoding(std::move(encoding)) {}

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   dim: int - the dimension of the encoding.
   */
  WeightedCategoricalBlock(uint32_t col, uint32_t dim, char delimiter=',')
      : _col(col), _encoding(std::make_shared<WeightedContiguousNumericId>(dim, delimiter)) {}

  uint32_t featureDim() const final { return _encoding->featureDim(); };

  bool isDense() const final { return _encoding->isDense(); };

  uint32_t expectedNumColumns() const final { return _col + 1; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    _encoding->encodeCategory(input_row.at(_col), vec);
  }

 private:
  uint32_t _col;
  std::shared_ptr<WeightedCategoricalEncoding> _encoding;
};

}  // namespace thirdai::dataset