#pragma once

#include <hashing/src/HashUtils.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Transformation.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace thirdai::data {

class HashPositionTransform final : public Transformation {
 public:
  HashPositionTransform(std::string input_column, std::string output_column,
                        uint32_t hash_range)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _dim(hash_range) {}

  ColumnMap apply(ColumnMap columns, State& state) const final {
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

 private:
  std::string _input_column;
  std::string _output_column;
  uint32_t _dim;
};

class OffsetPositionTransform final : public Transformation {
 public:
  OffsetPositionTransform(std::string input_column, std::string output_column,
                          uint32_t max_num_tokens)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _max_num_tokens(max_num_tokens) {}

  ColumnMap apply(ColumnMap columns, State& state) const final {
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
                      ArrayColumn<uint32_t>::make(
                          std::move(new_data), vocab_size * _max_num_tokens));
    return columns;
  }

 private:
  std::string _input_column;
  std::string _output_column;
  uint32_t _max_num_tokens;
};

}  // namespace thirdai::data