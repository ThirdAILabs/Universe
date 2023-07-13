#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

using dataset::TextEncoderPtr;
using dataset::TextTokenizerPtr;

class Text final : public Transformation {
 public:
  Text(std::string input_column, std::string output_column,
       TextTokenizerPtr tokenizer, TextEncoderPtr encoder,
       bool lowercase = false,
       size_t dim = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM);

  ColumnMap apply(ColumnMap columns) const final;

 private:
  std::string _input_column, _output_column;

  TextTokenizerPtr _tokenizer;
  TextEncoderPtr _encoder;

  bool _lowercase;
  size_t _dim;
};

}  // namespace thirdai::data