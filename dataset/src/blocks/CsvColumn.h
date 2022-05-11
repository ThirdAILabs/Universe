#pragma once

#include "../encodings/numstring/NumstringEncodingInterface.h"
#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/OneHotEncoding.h>
#include <dataset/src/utils/Conversions.h>
#include <memory>
#include <string_view>

namespace thirdai::dataset {

/**
 * A block for embedding a column that contains delimited strings.
 */
struct CsvColumnBlock : public Block {
  CsvColumnBlock(uint32_t col, std::shared_ptr<NumstringEncoding>& encoding, char delim=',')
      : _col(col), _delim(delim), _encoding(encoding) {}

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
    
    std::string_view numstr_arr(input_row[_col]);
    
    size_t start_pos = 0;
    while (start_pos != std::string_view::npos) {
      auto end_pos = numstr_arr.find(_delim, start_pos);
      _encoding->encodeNumstring(numstr_arr.substr(start_pos, end_pos), shared_feature_vector, idx_offset);  
    }
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
  uint32_t _col;
  char _delim;
  std::shared_ptr<NumstringEncoding> _encoding;
};

}  // namespace thirdai::dataset