#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/ContiguousNumericId.h>
#include <memory>
#include <optional>

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
  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEncoding> encoding, std::optional<char> delimiter)
      : _col(col), _encoding(std::move(encoding)), _delimiter(delimiter) {}

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   dim: int - the dimension of the encoding.
   */
  CategoricalBlock(uint32_t col, uint32_t dim)
      : _col(col), _encoding(std::make_shared<ContiguousNumericId>(dim)) {}

  uint32_t featureDim() const final { return _encoding->featureDim(); };

  bool isDense() const final { return _encoding->isDense(); };

  uint32_t expectedNumColumns() const final { return _col + 1; };

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    if (!_delimiter) {
      return _encoding->encodeCategory(input_row.at(_col), vec);
    }
    auto category_col = std::string(input_row.at(_col));
    auto categories = ProcessorUtils::parseCsvRow(category_col, _delimiter.value());
    for (auto& category : categories) {
      auto exception = _encoding->encodeCategory(category, vec);
      if (exception) {
        return exception;
      }
    }
    return nullptr;
  }

 private:
  uint32_t _col;
  std::shared_ptr<CategoricalEncoding> _encoding;
  std::optional<char> _delimiter;
};

}  // namespace thirdai::dataset