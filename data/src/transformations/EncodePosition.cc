#include "EncodePosition.h"
#include <hashing/src/HashUtils.h>
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
#pragma omp parallel for default(none) shared(input_column, hashed_tokens)
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

ColumnMap OffsetPositionTransform::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;
  auto input_column = columns.getArrayColumn<uint32_t>(_input_column);
  assertInputDimIsVocabSize(*input_column);

  std::vector<std::vector<uint32_t>> offset_tokens(input_column->numRows());
#pragma omp parallel for default(none) shared(input_column, offset_tokens)
  for (uint32_t i = 0; i < input_column->numRows(); i++) {
    offset_tokens[i].reserve(input_column->row(i).size());
    uint32_t pos = 0;
    for (uint32_t token : input_column->row(i)) {
      offset_tokens[i].push_back(encode(token, pos));
      ++pos;
    }
  }
  size_t dim = _vocab_size * _max_num_tokens;
  auto offset_col = ArrayColumn<uint32_t>::make(std::move(offset_tokens), dim);
  columns.setColumn(_output_column, offset_col);
  return columns;
}

uint32_t OffsetPositionTransform::encode(uint32_t token, uint32_t pos) const {
  uint32_t encoded_pos = std::min<uint32_t>(pos, _max_num_tokens - 1);
  return _vocab_size * encoded_pos + token;
}

void OffsetPositionTransform::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  auto output = apply(input, state);

  explainEncodedPositions(input, output, _input_column, _output_column,
                          explanations);
}
}  // namespace thirdai::data