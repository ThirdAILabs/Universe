#pragma once

#include <data/src/transformations/Transformation.h>
#include <string>

namespace thirdai::data {

class HashPositionTransform final : public Transformation {
 public:
  HashPositionTransform(std::string input_column, std::string output_column,
                        uint32_t hash_range)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _dim(hash_range) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

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

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column;
  std::string _output_column;
  uint32_t _max_num_tokens;
};

}  // namespace thirdai::data