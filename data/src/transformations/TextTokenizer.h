#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>

namespace thirdai::data {

class TextTokenizer final : public Transformation {
 public:
  TextTokenizer(
      std::string input_column, std::string output_column,
      dataset::TextTokenizerPtr tokenizer, dataset::TextEncoderPtr encoder,
      bool lowercase = false,
      size_t dim = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM);

  explicit TextTokenizer(const proto::data::TextTokenizer& text);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _input_column, _output_column;

  dataset::TextTokenizerPtr _tokenizer;
  dataset::TextEncoderPtr _encoder;

  bool _lowercase;
  size_t _dim;
};

}  // namespace thirdai::data