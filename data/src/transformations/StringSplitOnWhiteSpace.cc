#include "StringSplitOnWhiteSpace.h"
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <utils/text/StringManipulation.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/columns/ValueColumns.h>
#include <string>
#include <vector>

namespace thirdai::data {


ColumnMap StringSplitOnWhiteSpace::apply(ColumnMap columns, State& state) const {
  (void)state;

  auto input_column = columns.getValueColumn<std::string>(_input_column_name);
  size_t num_rows = columns.numRows();

  std::vector<std::vector<std::string>> split_results(num_rows);

#pragma omp parallel for default(none) shared(split_results, input_column, num_rows)
  for (size_t i = 0; i < num_rows; i++) {
    split_results[i] = text::splitOnWhiteSpace(input_column->value(i));
  }

  auto output_column = ArrayColumn<std::string>::make(std::move(split_results));
  columns.setColumn(_output_column_name, output_column); 

  return columns;
}

ar::ConstArchivePtr StringSplitOnWhiteSpace::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));
  map->set("input_column", ar::str(_input_column_name));
  map->set("output_column", ar::str(_output_column_name));
  return map;
}

StringSplitOnWhiteSpace::StringSplitOnWhiteSpace(const ar::Archive& archive)
    : _input_column_name(archive.str("input_column")),
      _output_column_name(archive.str("output_column_prefix")) {}

}  // namespace thirdai::data
