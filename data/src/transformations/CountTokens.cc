#include "CountTokens.h"
#include <data/src/columns/ValueColumns.h>

namespace thirdai::data {

thirdai::data::ColumnMap thirdai::data::CountTokens::apply(ColumnMap columns,
                                                           State& state) const {
  (void)state;
  auto tokens_column = columns.getArrayColumn<uint32_t>(_input_column);
  std::vector<uint32_t> num_tokens(tokens_column->numRows());

#pragma omp parallel for default(none) shared(tokens_column, num_tokens)
  for (uint32_t i = 0; i < tokens_column->numRows(); ++i) {
    num_tokens[i] = tokens_column->row(i).size();
    if (_ceiling && num_tokens[i] > _ceiling) {
      num_tokens[i] = *_ceiling;
    }
  }

  std::optional<uint32_t> dim =
      _ceiling ? std::make_optional(*_ceiling + 1) : std::nullopt;

  auto new_column = ValueColumn<uint32_t>::make(
      /* data= */ std::move(num_tokens), /* dim= */ dim);
  columns.setColumn(/* name= */ _output_column,
                    /* column= */ new_column);
  return columns;
}

}  // namespace thirdai::data
