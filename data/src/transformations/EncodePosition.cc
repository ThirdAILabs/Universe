#include "EncodePosition.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <hashing/src/HashUtils.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/rca/ExplanationMap.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace thirdai::data {

ColumnMap HashPositionTransform::apply(ColumnMap columns, State& state) const {
  (void)state;
  auto input_column = columns.getArrayColumn<uint32_t>(_input_column);
  std::vector<std::vector<uint32_t>> hashed_tokens(input_column->numRows());
#pragma omp parallel for default(none) \
    shared(input_column, hashed_tokens) if (columns.numRows() > 1)
  for (uint32_t i = 0; i < input_column->numRows(); i++) {
    hashed_tokens[i].reserve(input_column->row(i).size());
    uint32_t pos = 0;
    for (uint32_t token : input_column->row(i)) {
      uint32_t pos_encoded_token = hashing::combineHashes(pos, token) % _dim;
      hashed_tokens[i].push_back(pos_encoded_token);
      ++pos;
    }
  }
  auto hashed_col = ArrayColumn<uint32_t>::make(std::move(hashed_tokens), _dim);
  columns.setColumn(/* name= */ _output_column, hashed_col);
  return columns;
}

void explainEncodedPositions(const ColumnMap& input, const ColumnMap& output,
                             const std::string& input_column,
                             const std::string& output_column,
                             ExplanationMap& explanations) {
  auto input_tokens = input.getArrayColumn<uint32_t>(input_column)->row(0);
  auto encoded_tokens = output.getArrayColumn<uint32_t>(output_column)->row(0);

  for (size_t i = 0; i < encoded_tokens.size(); i++) {
    std::string explanation =
        explanations.explain(input_column, input_tokens[i]) + " at position " +
        std::to_string(i);
    explanations.store(output_column, encoded_tokens[i], explanation);
  }
}

void HashPositionTransform::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  auto output = apply(input, state);

  explainEncodedPositions(input, output, _input_column, _output_column,
                          explanations);
}

ar::ConstArchivePtr HashPositionTransform::toArchive() const {
  auto map = ar::Map::make();

  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));
  map->set("dim", ar::u64(_dim));

  return map;
}

template void HashPositionTransform::serialize(cereal::BinaryInputArchive&);
template void HashPositionTransform::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void HashPositionTransform::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_column, _dim);
}

ColumnMap OffsetPositionTransform::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;
  auto input_column = columns.getArrayColumn<uint32_t>(_input_column);

  if (!input_column->dim()) {
    throw std::invalid_argument(
        "OffsetPositionTransform: input column must have a dimension.");
  }
  uint32_t vocab_size = input_column->dim().value();

  std::vector<std::vector<uint32_t>> offset_tokens(input_column->numRows());
#pragma omp parallel for default(none) \
    shared(input_column, offset_tokens, vocab_size)
  for (uint32_t i = 0; i < input_column->numRows(); i++) {
    offset_tokens[i].reserve(input_column->row(i).size());
    uint32_t pos = 0;
    for (uint32_t token : input_column->row(i)) {
      uint32_t encoded_pos = std::min<uint32_t>(pos, _max_num_tokens - 1);
      uint32_t pos_encoded_token = vocab_size * encoded_pos + token;
      offset_tokens[i].push_back(pos_encoded_token);
      ++pos;
    }
  }
  size_t dim = vocab_size * _max_num_tokens;
  auto offset_col = ArrayColumn<uint32_t>::make(std::move(offset_tokens), dim);
  columns.setColumn(_output_column, offset_col);
  return columns;
}

void OffsetPositionTransform::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  auto output = apply(input, state);

  explainEncodedPositions(input, output, _input_column, _output_column,
                          explanations);
}

ar::ConstArchivePtr OffsetPositionTransform::toArchive() const {
  auto map = ar::Map::make();

  map->set("input_column", ar::str(_input_column));
  map->set("output_column", ar::str(_output_column));
  map->set("max_tokens", ar::u64(_max_num_tokens));

  return map;
}

template void OffsetPositionTransform::serialize(cereal::BinaryInputArchive&);
template void OffsetPositionTransform::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void OffsetPositionTransform::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this), _input_column,
          _output_column, _max_num_tokens);
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::HashPositionTransform)
CEREAL_REGISTER_TYPE(thirdai::data::OffsetPositionTransform)