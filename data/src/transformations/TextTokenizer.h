#pragma once

#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <memory>

namespace thirdai::data {

class TextTokenizer final : public Transformation {
 public:
  TextTokenizer(
      std::string input_column, std::string output_indices,
      std::optional<std::string> output_values,
      dataset::TextTokenizerPtr tokenizer, dataset::TextEncoderPtr encoder,
      bool lowercase = false,
      size_t dim = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM);

  static std::shared_ptr<TextTokenizer> make(
      std::string input_column, std::string output_indices,
      std::optional<std::string> output_values,
      dataset::TextTokenizerPtr tokenizer, dataset::TextEncoderPtr encoder,
      bool lowercase = false,
      size_t dim = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM) {
    return std::make_shared<TextTokenizer>(
        std::move(input_column), std::move(output_indices),
        std::move(output_values), std::move(tokenizer), std::move(encoder),
        lowercase, dim);
  }

  explicit TextTokenizer(const proto::data::TextTokenizer& text);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  static std::pair<std::vector<uint32_t>, std::vector<float>>
  deduplicateIndices(std::vector<uint32_t>&& tokens);

  std::string _input_column, _output_indices;
  std::optional<std::string> _output_values;

  dataset::TextTokenizerPtr _tokenizer;
  dataset::TextEncoderPtr _encoder;

  bool _lowercase;
  size_t _dim;
};

}  // namespace thirdai::data