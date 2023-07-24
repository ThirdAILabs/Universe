#include "StringConcat.h"
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <string>

namespace thirdai::data {

ColumnMap StringConcat::apply(ColumnMap columns, State& state) const {
  (void)state;

  std::vector<ValueColumnBasePtr<std::string>> input_columns;
  for (const auto& col_name : _input_column_names) {
    input_columns.push_back(columns.getValueColumn<std::string>(col_name));
  }

  std::vector<std::string> output(columns.numRows());

#pragma omp parallel for default(none) shared(output, columns, input_columns)
  for (size_t i = 0; i < columns.numRows(); i++) {
    std::string concat;
    for (const auto& col : input_columns) {
      concat.append(col->value(i)).push_back(' ');
    }
    output[i] = std::move(concat);
  }

  auto output_column = ValueColumn<std::string>::make(std::move(output));
  columns.setColumn(_output_column_name, output_column);

  return columns;
}

}  // namespace thirdai::data