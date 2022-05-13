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
   *   numerical_id: bool - whether the categorical feature is numerical. 
   *     Defaults to true.
   */
  CategoricalBlock(uint32_t col, std::shared_ptr<CategoricalEncoding>& encoding,
                   bool numerical_id = true)
      : _col(col), _numerical_id(numerical_id), _encoding(encoding) {}

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
  CategoricalBlock(uint32_t col, uint32_t dim, bool numerical_id = true)
      : _col(col),
        _numerical_id(numerical_id),
        _encoding(std::make_shared<OneHotEncoding>(dim)) {}

  uint32_t featureDim() final { return _encoding->featureDim(); };

  bool isDense() final { return _encoding->isDense(); };

 protected:
  void extendVector(const std::vector<std::string>& input_row,
                    ExtendableVector& vec) {
    
    const std::string& col_str = input_row[_col];
    char* end;
    uint32_t id;

    // Get numerical ID of categorical feature.
    // If ID is not originally a number, hash it to get one.
    if (_numerical_id) {
        id = std::strtoul(col_str.c_str(), &end, 10);
    } else {
        id = hashing::MurmurHash(col_str.c_str(), col_str.length(), 0);
    }
        
    _encoding->encodeCategory(id, vec);
  }

 private:
  uint32_t _col;
  bool _numerical_id;
  std::shared_ptr<CategoricalEncoding> _encoding;
};

}  // namespace thirdai::dataset