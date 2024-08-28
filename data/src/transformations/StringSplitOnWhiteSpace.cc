#include "StringSplitOnWhiteSpace.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <utils/text/StringManipulation.h>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::data {

ColumnMap StringSplitOnWhiteSpace::apply(ColumnMap columns,
                                         State& state) const {
  (void)state;

  auto input_column = columns.getValueColumn<std::string>(_input_column_name);
  size_t num_rows = columns.numRows();

  std::vector<std::vector<std::string>> split_results(num_rows);
  std::vector<std::vector<std::pair<size_t, size_t>>> offset_results(num_rows);

#pragma omp parallel for default(none) \
    shared(split_results, offset_results, input_column, num_rows)
  for (size_t i = 0; i < num_rows; i++) {
    auto [tokens, offsets] =
        StringSplitOnWhiteSpace::splitOnWhiteSpaceWithOffsets(
            input_column->value(i));
    split_results[i] = std::move(tokens);
    offset_results[i] = std::move(offsets);
  }

  auto output_column = ArrayColumn<std::string>::make(std::move(split_results));
  auto offset_column =
      ArrayColumn<std::pair<size_t, size_t>>::make(std::move(offset_results));

  columns.setColumn(_output_column_name, output_column);
  columns.setColumn(_output_column_name + "_offsets", offset_column);

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
      _output_column_name(archive.str("output_column")) {}

}  // namespace thirdai::data
