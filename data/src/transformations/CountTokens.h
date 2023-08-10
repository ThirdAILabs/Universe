#pragma once

#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Transformation.h>
#include <algorithm>
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

  ColumnMap apply(ColumnMap columns, State& state) const final {
    (void)state;
    auto tokens_column = columns.getArrayColumn<uint32_t>(_input_column);
    std::vector<uint32_t> new_data(tokens_column->numRows());

#pragma omp parallel for default(none) shared(tokens_column, new_data)
    for (uint32_t i = 0; i < tokens_column->numRows(); ++i) {
      new_data[i] = tokens_column->row(i).size();
      if (_ceiling && new_data[i] > _ceiling) {
        new_data[i] = _ceiling.value();
      }
    }

    std::optional<uint32_t> dim =
        _ceiling ? std::make_optional(*_ceiling + 1) : std::nullopt;

    auto new_column = ValueColumn<uint32_t>::make(
        /* data= */ std::move(new_data), /* dim= */ dim);
    columns.setColumn(/* name= */ _output_column,
                      /* column= */ new_column);
    return columns;
  }

 private:
  std::string _input_column;
  std::string _output_column;
  std::optional<uint32_t> _ceiling;
};

}  // namespace thirdai::data