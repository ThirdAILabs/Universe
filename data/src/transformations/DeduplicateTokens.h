#pragma once

#include <data/src/transformations/Transformation.h>
#include <optional>
#include <string>

namespace thirdai::data {

/**
 * Deduplicates indices in each row, summing up their values if provided or
 * the frequency otherwise.
 * For example, given a row of indices [0, 0, 1, 0], the transformation produces
 * a row of indices [0, 1] and a row of values [3, 1]. In this example, we don't
 * provide the values so the output values represent the frequencies of the
 * indices.
 * Given a row of indices [0, 0, 1, 0] and values [1.0, 2.0, 3.0, 4.0], the
 * transformation produces a row of indices [0, 1] and a row of values
 * [7.0, 3.0].
 */
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

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "dedup_tokens"; }

 private:
  std::string _input_indices_column;
  std::optional<std::string> _input_values_column;
  std::string _output_indices_column;
  std::string _output_values_column;
};

}  // namespace thirdai::data