#include "TextTokenizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <data/src/columns/VectorColumns.h>

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

ColumnMap TextTokenizer::apply(ColumnMap columns) const {
  StringColumnPtr text_col = columns.getStringColumn(_input_column);

  std::vector<std::vector<uint32_t>> output_tokens(text_col->numRows());

#pragma omp parallel for default(none) shared(text_col, output_tokens)
  for (size_t i = 0; i < text_col->numRows(); i++) {
    std::string string = text_col->at(i);

    if (_lowercase) {
      string = text::lower(string);
    }

    std::vector<uint32_t> tokens = _tokenizer->tokenize(string);
    std::vector<uint32_t> indices = _encoder->encode(tokens);
    dataset::token_encoding::mod(indices, _dim);

    output_tokens[i] = std::move(indices);
  }

  auto token_col =
      std::make_shared<CppTokenArrayColumn>(std::move(output_tokens), _dim);

  columns.setColumn(_output_column, token_col);
  return columns;
}

template void TextTokenizer::serialize(cereal::BinaryInputArchive&);
template void TextTokenizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TextTokenizer::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_column, _tokenizer, _encoder, _lowercase, _dim);
}

}  // namespace thirdai::data