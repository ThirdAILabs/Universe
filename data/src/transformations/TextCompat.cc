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
                       size_t encoding_dim, size_t hash_range)
    : _input_column(std::move(input_column)),
      _output_indices(std::move(output_indices)),
      _output_values(std::move(output_values)),
      _tokenizer(std::move(tokenizer)),
      _encoder(std::move(encoder)),
      _lowercase(lowercase),
      _encoding_dim(encoding_dim),
      _hash_range(hash_range) {}

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

    dataset::token_encoding::mod(tokens, _encoding_dim);

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
      ArrayColumn<uint32_t>::make(std::move(output_indices), _hash_range);
  columns.setColumn(_output_indices, indices_col);

  auto values_col = ArrayColumn<float>::make(std::move(output_values));
  columns.setColumn(_output_values, values_col);

  return columns;
}

ar::ConstArchivePtr TextCompat::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column));
  map->set("output_indices", ar::str(_output_indices));
  map->set("output_values", ar::str(_output_values));

  map->set("tokenizer", _tokenizer->toArchive());
  map->set("encoder", _encoder->toArchive());
  map->set("lowercase", ar::boolean(_lowercase));
  map->set("encoding_dim", ar::u64(_encoding_dim));
  map->set("hash_range", ar::u64(_hash_range));

  return map;
}

TextCompat::TextCompat(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _output_indices(archive.str("output_indices")),
      _output_values(archive.str("output_values")),
      _tokenizer(
          dataset::TextTokenizer::fromArchive(*archive.get("tokenizer"))),
      _encoder(dataset::TextEncoder::fromArchive(*archive.get("encoder"))),
      _lowercase(archive.getAs<ar::Boolean>("lowercase")),
      _encoding_dim(archive.u64("encoding_dim")),
      _hash_range(archive.u64("hash_range")) {}

template void TextCompat::serialize(cereal::BinaryInputArchive&);
template void TextCompat::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TextCompat::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_indices, _output_values, _tokenizer, _encoder, _lowercase,
          _encoding_dim, _hash_range);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TextCompat)