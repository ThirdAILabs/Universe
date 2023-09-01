#pragma once

#include <data/src/transformations/Transformation.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::data {

/**
 * Counts the number of tokens in each row of a token array column.
 */
class CountTokens final : public Transformation {
 public:
  CountTokens(std::string input_column, std::string output_column,
              std::optional<uint32_t> max_tokens)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _max_tokens(max_tokens) {}

  static std::shared_ptr<CountTokens> make(std::string input_column,
                                           std::string output_column,
                                           std::optional<uint32_t> max_tokens) {
    return std::make_shared<CountTokens>(std::move(input_column),
                                         std::move(output_column), max_tokens);
  }

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column;
  std::string _output_column;
  std::optional<uint32_t> _max_tokens;
};

}  // namespace thirdai::data