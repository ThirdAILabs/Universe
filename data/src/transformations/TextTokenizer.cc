#include "TextTokenizer.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/ArrayColumns.h>
#include <utils/text/StringManipulation.h>
#include <string>
#include <tuple>
#include <vector>

namespace thirdai::data {

TextTokenizer::TextTokenizer(std::string input_column,
                             std::string output_indices,
                             std::optional<std::string> output_values,
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

ColumnMap TextTokenizer::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto text_col = columns.getValueColumn<std::string>(_input_column);

  std::vector<std::vector<uint32_t>> output_indices(text_col->numRows());

  std::vector<std::vector<float>> output_values;
  if (_output_values) {
    output_values.assign(text_col->numRows(), {});
  }

#pragma omp parallel for default(none) \
    shared(text_col, output_indices, output_values) if (columns.numRows() > 1)
  for (size_t i = 0; i < text_col->numRows(); i++) {
    std::string string = text_col->value(i);

    if (_lowercase) {
      string = text::lower(string);
    }

    std::vector<uint32_t> tokens = _tokenizer->tokenize(string);
    std::vector<uint32_t> indices = _encoder->encode(tokens);
    dataset::token_encoding::mod(indices, _dim);

    if (_output_values) {
      // Deduplicating indices can provide a speedup on longer texts, this
      // method also has the added benefit of sorting the indices for better
      // memory access patterns too.
      auto [dedup_indices, dedup_values] =
          deduplicateIndices(std::move(indices));
      output_indices[i] = std::move(dedup_indices);
      output_values[i] = std::move(dedup_values);
    } else {
      std::sort(indices.begin(), indices.end());
      output_indices[i] = std::move(indices);
    }
  }

  auto indices_col =
      ArrayColumn<uint32_t>::make(std::move(output_indices), _dim);
  columns.setColumn(_output_indices, indices_col);

  if (_output_values) {
    auto values_col = ArrayColumn<float>::make(std::move(output_values));
    columns.setColumn(*_output_values, values_col);
  }

  return columns;
}

void TextTokenizer::buildExplanationMap(const ColumnMap& input, State& state,
                                        ExplanationMap& explanations) const {
  (void)state;

  const std::string& input_text =
      input.getValueColumn<std::string>(_input_column)->value(0);

  std::string to_tokenize = _lowercase ? text::lower(input_text) : input_text;

  std::vector<uint32_t> tokens = _tokenizer->tokenize(to_tokenize);
  std::vector<uint32_t> indices = _encoder->encode(tokens);
  dataset::token_encoding::mod(indices, _dim);

  for (const auto& index : indices) {
    uint32_t token = _encoder->undoEncoding(tokens, index, _dim);
    auto word = _tokenizer->getResponsibleWord(to_tokenize, token);

    explanations.store(_output_indices, index,
                       "word '" + word + "' from " +
                           explanations.explain(_input_column, input_text));
  }
}

std::pair<std::vector<uint32_t>, std::vector<float>>
TextTokenizer::deduplicateIndices(std::vector<uint32_t>&& tokens) {
  if (tokens.empty()) {
    return {{}, {}};
  }

  std::sort(tokens.begin(), tokens.end());

  std::vector<uint32_t> indices;
  std::vector<float> values;

  uint32_t curr_token = tokens.front();
  float count = 0.0;
  for (uint32_t token : tokens) {
    if (token == curr_token) {
      count++;
    } else {
      indices.push_back(curr_token);
      values.push_back(count);
      curr_token = token;
      count = 1.0;
    }
  }
  indices.push_back(curr_token);
  values.push_back(count);

  return {std::move(indices), std::move(values)};
}

ar::ConstArchivePtr TextTokenizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column));
  map->set("output_indices", ar::str(_output_indices));
  if (_output_values) {
    map->set("output_values", ar::str(*_output_values));
  }
  map->set("tokenizer", _tokenizer->toArchive());
  map->set("encoder", _encoder->toArchive());
  map->set("lowercase", ar::boolean(_lowercase));
  map->set("dim", ar::u64(_dim));

  return map;
}

TextTokenizer::TextTokenizer(const ar::Archive& archive)
    : _input_column(archive.str("input_column")),
      _output_indices(archive.str("output_indices")),
      _output_values(archive.getOpt<ar::Str>("output_values")),
      _tokenizer(
          dataset::TextTokenizer::fromArchive(*archive.get("tokenizer"))),
      _encoder(dataset::TextEncoder::fromArchive(*archive.get("encoder"))),
      _lowercase(archive.getAs<ar::Boolean>("lowercase")),
      _dim(archive.u64("dim")) {}

template void TextTokenizer::serialize(cereal::BinaryInputArchive&);
template void TextTokenizer::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void TextTokenizer::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_indices, _output_values, _tokenizer, _encoder, _lowercase,
          _clean_text, _dim);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::TextTokenizer)