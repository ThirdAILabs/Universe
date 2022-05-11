#pragma once

#include "../encodings/array/ArrayEncodingInterface.h"
#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/OneHotEncoding.h>
#include <dataset/src/utils/Conversions.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace thirdai::dataset {

/**
 * A block for embedding multiple columns as a single array.
 */
struct MultiColumnArrayBlock : public Block {
  MultiColumnArrayBlock(uint32_t start_col, std::shared_ptr<ArrayEncoding>& encoding, uint32_t end_col=0)
      : _start_col(start_col), _end_col(end_col), _encoding(encoding) {}

  /**
   * Extracts features from input row and adds it to shared feature vector.
   *
   * Arguments:
   * input_row: a list of columns for a single row.
   * shared_feature_vector: a vector that is shared among all blocks operating
   * on a particular row. This make it easier for the pipeline object to
   *   concatenate the features produced by each block.
   * idx_offset: the offset to shift the feature indices by if the preceeding
   *   section of the output vector is occupied by other features.
   */
  void process(const std::vector<std::string>& input_row,
               BuilderVector& shared_feature_vector,
               uint32_t idx_offset) final {
    
    if (_end_col > input_row.size()) {
      std::stringstream ss;
      ss << "[MultiColumnArray] Given end_col = " << _end_col << " but row only has " << input_row.size() << " columns.";
      throw std::invalid_argument(ss.str());
    }

    auto end_col = _end_col == 0 
      ? input_row.size()
      : _end_col;

    uint32_t i = _start_col;
    
    _encoding->encodeArray([&]() -> std::optional<std::string> {
      if (i < end_col) {
        const auto& str = input_row[i];
        ++i;
        return {str};
      } 
      return {};
    }, shared_feature_vector, idx_offset);
  };

  /**
   * Returns the dimension of extracted features.
   * This is needed when composing different features into a single vector.
   */
  uint32_t featureDim() final { return _encoding->featureDim(); };

  /**
   * True if the block produces dense features, False otherwise.
   */
  bool isDense() final { return _encoding->isDense(); };

 private:
  
  uint32_t _start_col;
  uint32_t _end_col;
  std::shared_ptr<ArrayEncoding> _encoding;
};

}  // namespace thirdai::dataset