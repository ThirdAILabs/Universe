#include "EncodePosition.h"
#include <hashing/src/HashUtils.h>
#include <data/src/columns/ArrayColumns.h>
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
      uint32_t pos_encoded_token = vocab_size * pos + token;
      offset_tokens[i].push_back(pos_encoded_token);
      ++pos;
    }
  }
  size_t dim = vocab_size * _max_num_tokens;
  auto offset_col = ArrayColumn<uint32_t>::make(std::move(offset_tokens), dim);
  columns.setColumn(_output_column, offset_col);
  return columns;
}
}  // namespace thirdai::data