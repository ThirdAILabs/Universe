#include "StringConcat.h"
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <string>

namespace thirdai::data {

StringConcat::StringConcat(const proto::data::StringConcat& string_concat)
    : _input_column_names(string_concat.input_columns().begin(),
                          string_concat.input_columns().end()),
      _output_column_name(string_concat.output_column()),
      _seperator(string_concat.seperator()) {}

ColumnMap StringConcat::apply(ColumnMap columns, State& state) const {
  (void)state;

  std::vector<ValueColumnBasePtr<std::string>> input_columns;
  for (const auto& col_name : _input_column_names) {
    input_columns.push_back(columns.getValueColumn<std::string>(col_name));
  }

  std::vector<std::string> output(columns.numRows());

#pragma omp parallel for default(none) \
    shared(output, columns, input_columns) if (columns.numRows() > 1)
  for (size_t i = 0; i < columns.numRows(); i++) {
    std::string concat;
    for (size_t col_idx = 0; col_idx < input_columns.size(); col_idx++) {
      if (col_idx > 0) {
        concat.append(_separator);
      }
      concat.append(input_columns[col_idx]->value(i));
    }
    output[i] = std::move(concat);
  }

  auto output_column = ValueColumn<std::string>::make(std::move(output));
  columns.setColumn(_output_column_name, output_column);

  return columns;
}

proto::data::Transformation* StringConcat::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* string_concat = transformation->mutable_string_concat();

  *string_concat->mutable_input_columns() = {_input_column_names.begin(),
                                             _input_column_names.end()};
  string_concat->set_output_column(_output_column_name);
  string_concat->set_seperator(_seperator);

  return transformation;
}

}  // namespace thirdai::data