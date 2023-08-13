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
  std::vector<std::vector<uint32_t>> new_data(input_column->numRows());
#pragma omp parallel for default(none) shared(input_column, new_data)
  for (uint32_t i = 0; i < input_column->numRows(); ++i) {
    new_data[i].reserve(input_column->row(i).size());
    uint32_t pos = 0;
    for (uint32_t token : input_column->row(i)) {
      uint32_t pos_encoded_token = hashing::combineHashes(pos, token) % _dim;
      new_data[i].push_back(pos_encoded_token);
      ++pos;
    }
  }
  columns.setColumn(_output_column,
                    ArrayColumn<uint32_t>::make(std::move(new_data), _dim));
  return columns;
}

ColumnMap OffsetPositionTransform::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;
  auto input_column = columns.getArrayColumn<uint32_t>(_input_column);

  if (!input_column->dimension()) {
    throw std::invalid_argument(
        "OffsetPositionTransform: input column must have a dimension.");
  }
  uint32_t vocab_size = input_column->dimension()->dim;

  std::vector<std::vector<uint32_t>> new_data(input_column->numRows());
#pragma omp parallel for default(none) \
    shared(input_column, new_data, vocab_size)
  for (uint32_t i = 0; i < input_column->numRows(); ++i) {
    new_data[i].reserve(input_column->row(i).size());
    uint32_t pos = 0;
    for (uint32_t token : input_column->row(i)) {
      uint32_t pos_encoded_token = vocab_size * pos + token;
      new_data[i].push_back(pos_encoded_token);
      ++pos;
    }
  }
  columns.setColumn(_output_column,
                    ArrayColumn<uint32_t>::make(std::move(new_data),
                                                vocab_size * _max_num_tokens));
  return columns;
}
}  // namespace thirdai::data