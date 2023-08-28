#include "EncodePosition.h"
#include <hashing/src/HashUtils.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/rca/ExplanationMap.h>
#include <proto/sequence.pb.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace thirdai::data {

HashPositionTransform::HashPositionTransform(
    const proto::data::HashedPositionEncoding& hash_position)
    : _input_column(hash_position.input_column()),
      _output_column(hash_position.output_column()),
      _dim(hash_position.dim()) {}

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

proto::data::Transformation* HashPositionTransform::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* hash_position = transformation->mutable_hashed_position_encoding();

  hash_position->set_input_column(_input_column);
  hash_position->set_output_column(_output_column);
  hash_position->set_dim(_dim);

  return transformation;
}

OffsetPositionTransform::OffsetPositionTransform(
    const proto::data::OffsetPositionEncoding& offset_position)
    : _input_column(offset_position.input_column()),
      _output_column(offset_position.output_column()),
      _max_num_tokens(offset_position.max_tokens()) {}

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

proto::data::Transformation* OffsetPositionTransform::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* hash_position = transformation->mutable_offset_position_encoding();

  hash_position->set_input_column(_input_column);
  hash_position->set_output_column(_output_column);
  hash_position->set_max_tokens(_max_num_tokens);

  return transformation;
}

}  // namespace thirdai::data
