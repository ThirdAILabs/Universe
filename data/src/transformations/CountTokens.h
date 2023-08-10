#pragma once

#include <data/src/transformations/Transformation.h>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace thirdai::data {

class CountTokens final : public Transformation {
 public:
  CountTokens(std::string input_column, std::string output_column,
              std::optional<uint32_t> ceiling)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)),
        _ceiling(ceiling) {}

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  std::string _input_column;
  std::string _output_column;
  std::optional<uint32_t> _ceiling;
};

}  // namespace thirdai::data