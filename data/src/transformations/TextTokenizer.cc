#include "TextTokenizer.h"
#include <data/src/columns/ArrayColumns.h>

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

#pragma omp parallel for default(none) shared(text_col, output_tokens)
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

proto::data::Transformation* TextTokenizer::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* text = transformation->mutable_text_tokenizer();

  text->set_input_column(_input_column);
  text->set_output_column(_output_column);
  text->set_allocated_tokenizer(_tokenizer->toProto());
  text->set_allocated_encoder(_encoder->toProto());
  text->set_lowercase(_lowercase);
  text->set_dim(_dim);

  return transformation;
}

}  // namespace thirdai::data
