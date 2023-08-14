#pragma once

#include <data/src/transformations/Transformation.h>
#include <optional>
#include <string>

namespace thirdai::data {

class DeduplicateTokens final : public Transformation {
 public:
  DeduplicateTokens(std::string input_indices_column,
                    std::optional<std::string> input_values_column,
                    std::string output_indices_column,
                    std::string output_values_column)
      : _input_indices_column(std::move(input_indices_column)),
        _input_values_column(std::move(input_values_column)),
        _output_indices_column(std::move(output_indices_column)),
        _output_values_column(std::move(output_values_column)) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_indices_column;
  std::optional<std::string> _input_values_column;
  std::string _output_indices_column;
  std::string _output_values_column;
};

}  // namespace thirdai::data