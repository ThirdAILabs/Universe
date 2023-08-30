#include "TextTokenizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <data/src/columns/ArrayColumns.h>
#include <string>

namespace thirdai::data {

TextTokenizer::TextTokenizer(std::string input_column,
                             std::string output_column,
                             dataset::TextTokenizerPtr tokenizer,
                             dataset::TextEncoderPtr encoder, bool lowercase,
                             size_t dim)
    : _input_column(std::move(input_column)),
      _output_column(std::move(output_column)),
      _tokenizer(std::move(tokenizer)),
      _encoder(std::move(encoder)),
      _lowercase(lowercase),
      _dim(dim) {}

ColumnMap TextTokenizer::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_col = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> output_tokens(text_col->numRows());

#pragma omp parallel for default(none) \
    shared(text_col, output_tokens) if (columns.numRows() > 1)
  for (size_t i = 0; i < text_col->numRows(); i++) {
    std::string string = text_col->value(i);

    if (_lowercase) {
      string = text::lower(string);
    }

    std::vector<uint32_t> tokens = _tokenizer->tokenize(string);
    std::vector<uint32_t> indices = _encoder->encode(tokens);
    dataset::token_encoding::mod(indices, _dim);

    output_tokens[i] = std::move(indices);
  }

  auto token_col = ArrayColumn<uint32_t>::make(std::move(output_tokens), _dim);

  columns.setColumn(_output_column, token_col);
  return columns;
}

void TextTokenizer::buildExplanationMap(const ColumnMap& input, State& state,
                                        ExplanationMap& explanations) const {
  (void)state;

  const std::string& text =
      input.getValueColumn<std::string>(_input_column)->value(0);

  std::vector<uint32_t> tokens = _tokenizer->tokenize(text);
  std::vector<uint32_t> indices = _encoder->encode(tokens);
  dataset::token_encoding::mod(indices, _dim);

  for (const auto& index : indices) {
    uint32_t token = _encoder->undoEncoding(tokens, index, _dim);
    auto word = _tokenizer->getResponsibleWord(text, token);

    explanations.store(_output_column, index,
                       "word '" + word + "' from " +
                           explanations.explain(_input_column, text));
  }
}

template void TextTokenizer::serialize(cereal::BinaryInputArchive&);
template void TextTokenizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TextTokenizer::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_column, _tokenizer, _encoder, _lowercase, _dim);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TextTokenizer)