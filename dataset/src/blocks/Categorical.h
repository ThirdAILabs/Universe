#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/categorical/CategoricalEncodingInterface.h>
#include <dataset/src/encodings/categorical/OneHotEncoding.h>
#include <hashing/src/MurmurHash.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an 
 * identification string).
 */
struct CategoricalBlock : public Block {

  /**
   * Constructor.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   encoding: CategoricalEncoding - the categorical feature encoding model.
   */
  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEncoding>& encoding,
                   bool from_string = false)
      : _col(col), _from_string(from_string), _encoding(encoding) {}

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the categorical feature to be encoded.
   *   dim: int - the dimension of the encoding.
   */
  CategoricalBlock(uint32_t col, uint32_t dim, bool from_string = false)
      : _col(col),
        _from_string(from_string),
        _encoding(std::make_shared<OneHotEncoding>(dim)) {}

  uint32_t featureDim() final { return _encoding->featureDim(); };

  bool isDense() final { return _encoding->isDense(); };

 protected:
  void extendVector(const std::vector<std::string>& input_row,
                    ExtendableVector& vec) {
    
    const std::string& col_str = input_row[_col];
    char* end;
    uint32_t id;

    // Get ID of categorical feature.
    // If ID is not originally a number, hash it to get one.
    if (_from_string) {
        id = hashing::MurmurHash(col_str.c_str(), col_str.length(), 0);
    } else {
        id = std::strtoul(col_str.c_str(), &end, 10);
    }
        
    _encoding->encodeCategory(id, vec);
  }

 private:
  uint32_t _col;
  bool _from_string;
  std::shared_ptr<CategoricalEncoding> _encoding;
};

}  // namespace thirdai::dataset