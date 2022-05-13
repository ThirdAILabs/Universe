#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/text/UniGram.h>
#include <dataset/src/encodings/text/TextEncodingInterface.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
struct TextBlock : public Block {
  /**
   * Constructor.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the text to be encoded.
   *   encoding: TextEncoding - the text encoding model.
   */
  TextBlock(uint32_t col, std::shared_ptr<TextEncoding>& encoding)
      : _col(col), _encoding(encoding) {}

  /**
   * Constructor with default encoder.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the text to be encoded.
   *   dim: int - the dimension of the encoding.
   */
  TextBlock(uint32_t col, uint32_t dim)
      : _col(col), _encoding(std::make_shared<UniGram>(dim)) {}

  uint32_t featureDim() final { return _encoding->featureDim(); };

  bool isDense() final { return _encoding->isDense(); };

 protected:
  void buildExtension(const std::vector<std::string>& input_row,
                      ExtendableVector& vec) final {
    _encoding->encodeText(input_row[_col], vec);
  }

 private:
  uint32_t _col;
  std::shared_ptr<TextEncoding> _encoding;
};

}  // namespace thirdai::dataset