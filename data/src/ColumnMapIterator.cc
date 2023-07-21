#include "ColumnMapIterator.h"
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/utils/CsvParser.h>

namespace thirdai::data {

ColumnMapIterator::ColumnMapIterator(DataSourcePtr data_source, char delimiter,
                                     size_t chunk_size)
    : _data_source(std::move(data_source)),
      _delimiter(delimiter),
      _chunk_size(chunk_size) {
  auto header = _data_source->nextLine();
  if (!header.has_value()) {
    throw std::invalid_argument("DataSource was found to be empty.");
  }
  _column_names = dataset::parsers::CSV::parseLine(*header, _delimiter);
}

std::optional<ColumnMap> ColumnMapIterator::next() {
  auto rows = _data_source->nextBatch(_chunk_size);
  if (!rows) {
    return std::nullopt;
  }

  std::vector<std::vector<std::string>> columns(_column_names.size());

  // TODO(Nicholas): Should this be parallelized?
  for (const auto& row : *rows) {
    auto row_columns = dataset::parsers::CSV::parseLine(row, _delimiter);
    if (row_columns.size() != _column_names.size()) {
      throw std::invalid_argument(
          "Expected " + std::to_string(_column_names.size()) +
          " columns. But received row '" + row + "' with " +
          std::to_string(row.size()) + " columns.");
    }

    for (size_t i = 0; i < columns.size(); i++) {
      columns[i].push_back(row_columns[i]);
    }
  }

  return makeColumnMap(std::move(columns));
}

ColumnMap ColumnMapIterator::emptyColumnMap() const {
  return makeColumnMap(
      std::vector<std::vector<std::string>>(_column_names.size()));
}

ColumnMap ColumnMapIterator::makeColumnMap(
    std::vector<std::vector<std::string>>&& columns) const {
  std::unordered_map<std::string, ColumnPtr> column_map;

  for (size_t i = 0; i < columns.size(); i++) {
    column_map[_column_names[i]] =
        ValueColumn<std::string>::make(std::move(columns[i]));
  }

  return ColumnMap(std::move(column_map));
}

}  // namespace thirdai::data