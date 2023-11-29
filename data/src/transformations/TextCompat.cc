#include "TextCompat.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <string>
#include <tuple>
#include <vector>

namespace thirdai::data {

TextCompat::TextCompat(std::string input_column, std::string output_indices,
                       std::string output_values,
                       dataset::TextTokenizerPtr tokenizer,
                       dataset::TextEncoderPtr encoder, bool lowercase,
                       size_t dim)
    : _input_column(std::move(input_column)),
      _output_indices(std::move(output_indices)),
      _output_values(std::move(output_values)),
      _tokenizer(std::move(tokenizer)),
      _encoder(std::move(encoder)),
      _lowercase(lowercase),
      _dim(dim) {}

ColumnMap TextCompat::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_col = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> output_indices(text_col->numRows());

  std::vector<std::vector<float>> output_values(text_col->numRows());

#pragma omp parallel for default(none) \
    shared(text_col, output_indices, output_values) if (columns.numRows() > 1)
  for (size_t i = 0; i < text_col->numRows(); i++) {
    std::string string = text_col->value(i);

    if (_lowercase) {
      string = text::lower(string);
    }

    std::vector<uint32_t> tokens =
        _encoder->encode(_tokenizer->tokenize(string));

    std::vector<uint32_t> indices;
    std::vector<float> values;
    for (const auto& [index, value] :
         dataset::token_encoding::sumRepeatedIndices(tokens)) {
      indices.push_back(mimicHashedFeatureVector(index));
      values.push_back(value);
    }

    output_indices[i] = std::move(indices);
    output_values[i] = std::move(values);
  }

  auto indices_col =
      ArrayColumn<uint32_t>::make(std::move(output_indices), _dim);
  columns.setColumn(_output_indices, indices_col);

  auto values_col = ArrayColumn<float>::make(std::move(output_values));
  columns.setColumn(_output_values, values_col);

  return columns;
}

void TextCompat::buildExplanationMap(const ColumnMap& input, State& state,
                                     ExplanationMap& explanations) const {
  (void)state;

  const std::string& text =
      input.getValueColumn<std::string>(_input_column)->value(0);

  std::vector<uint32_t> tokens = _tokenizer->tokenize(text);
  std::vector<uint32_t> indices = _encoder->encode(tokens);

  for (const auto& index : indices) {
    uint32_t token = _encoder->undoEncoding(tokens, index, _dim);
    auto word = _tokenizer->getResponsibleWord(text, token);

    explanations.store(_output_indices, mimicHashedFeatureVector(index),
                       "word '" + word + "' from " +
                           explanations.explain(_input_column, text));
  }
}

template void TextCompat::serialize(cereal::BinaryInputArchive&);
template void TextCompat::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TextCompat::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_indices, _output_values, _tokenizer, _encoder, _lowercase,
          _dim);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TextCompat)