#pragma once

#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/ContiguousNumericId.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an
 * identification string).
 */
class CategoricalBlock : public Block {
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
  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEncoding> encoding)
      : _col(col), _encoding(std::move(encoding)) {}

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   dim: int - the dimension of the encoding.
   *   numerical_id: bool - whether the categorical feature is numerical.
   *     Defaults to true.
   */
  CategoricalBlock(uint32_t col, uint32_t dim)
      : _col(col), _encoding(std::make_shared<ContiguousNumericId>(dim)) {}

  uint32_t featureDim() const final { return _encoding->featureDim(); };

  bool isDense() const final { return _encoding->isDense(); };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                      SegmentedFeatureVector& vec) final {
    _encoding->encodeCategory(input_row.at(_col), vec);
  }

 private:
  uint32_t _col;
  std::shared_ptr<CategoricalEncoding> _encoding;
};

}  // namespace thirdai::dataset