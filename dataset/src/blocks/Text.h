#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/text/TextEncodingInterface.h>
#include <dataset/src/encodings/text/UniGram.h>
#include <memory>

namespace thirdai::dataset {

/**
 * A block that encodes text (e.g. sentences / paragraphs).
 */
class TextBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * Arguments:
   *   col: int - the column number of the input row containing
   *     the text to be encoded.
   *   encoding: TextEncoding - the text encoding model.
   */
  TextBlock(uint32_t col, std::shared_ptr<TextEncoding> encoding)
      : _col(col), _encoding(std::move(encoding)) {}

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

  uint32_t featureDim() const final { return _encoding->featureDim(); };

  bool isDense() const final { return _encoding->isDense(); };

  uint32_t expectedNumColumns() const final { return _col + 1; };

  std::string giveMessage(float gradient_ratio_value,std::unordered_map<uint32_t, std::string> col_num_col_name_map,float row_ratio_sum) const final {
    std::string message = "The Text column  "+col_num_col_name_map.at(_col)+" is  "+std::to_string(((gradient_ratio_value)/(row_ratio_sum))*100)+"% responsible.";
    return message;
  }

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    _encoding->encodeText(input_row.at(_col), vec);
  }

 private:
  uint32_t _col;
  std::shared_ptr<TextEncoding> _encoding;
};

}  // namespace thirdai::dataset