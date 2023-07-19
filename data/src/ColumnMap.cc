#include "ColumnMap.h"
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

ColumnMap::ColumnMap(std::unordered_map<std::string, ColumnPtr> columns)
    : _columns(std::move(columns)) {
  if (_columns.empty()) {
    throw std::invalid_argument(
        "Cannot construct ColumnMap from empty set of columns.");
  }

  std::optional<size_t> num_rows = std::nullopt;
  for (auto& [_, column] : _columns) {
    if (num_rows && column->numRows() != num_rows.value()) {
      throw std::invalid_argument(
          "All columns must have the same number of rows.");
    }
    num_rows = column->numRows();
  }
  _num_rows = num_rows.value();
}

template <typename T>
ArrayColumnBasePtr<T> ColumnMap::getArrayColumn(const std::string& name) const {
  auto column = std::dynamic_pointer_cast<ArrayColumnBase<T>>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to ArrayColumn.");
  }
  return column;
}

template ArrayColumnBasePtr<uint32_t> ColumnMap::getArrayColumn(
    const std::string&) const;
template ArrayColumnBasePtr<float> ColumnMap::getArrayColumn(
    const std::string&) const;
template ArrayColumnBasePtr<std::string> ColumnMap::getArrayColumn(
    const std::string&) const;

template <typename T>
ValueColumnBasePtr<T> ColumnMap::getValueColumn(const std::string& name) const {
  auto column = std::dynamic_pointer_cast<ValueColumnBase<T>>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to ValueColumn.");
  }
  return column;
}

template ValueColumnBasePtr<uint32_t> ColumnMap::getValueColumn(
    const std::string&) const;
template ValueColumnBasePtr<float> ColumnMap::getValueColumn(
    const std::string&) const;
template ValueColumnBasePtr<std::string> ColumnMap::getValueColumn(
    const std::string&) const;
template ValueColumnBasePtr<int64_t> ColumnMap::getValueColumn(
    const std::string&) const;

ColumnPtr ColumnMap::getColumn(const std::string& name) const {
  if (!_columns.count(name)) {
    throw std::invalid_argument("Unable to find column with name '" + name +
                                "'.");
  }
  return _columns.at(name);
}

void ColumnMap::setColumn(const std::string& name, ColumnPtr column) {
  // _columns.begin() is safe because the constructor to ColumnMap throws if the
  // supplied set of columns is empty.
  if (column->numRows() != _columns.begin()->second->numRows()) {
    throw std::invalid_argument(
        "Cannot insert a Column with a different number of rows into a "
        "ColumnMap.");
  }
  _columns[name] = std::move(column);
}

std::vector<std::string> ColumnMap::columns() const {
  std::vector<std::string> columns;
  for (auto const& map_entry : _columns) {
    columns.push_back(map_entry.first);
  }
  return columns;
}

void ColumnMap::shuffle(uint32_t seed) {
  std::vector<size_t> permutation(numRows());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(), std::mt19937{seed});

  for (auto& [_, column] : _columns) {
    column->shuffle(permutation);
  }
}

ColumnMap ColumnMap::concat(ColumnMap& other) {
  std::unordered_map<std::string, ColumnPtr> new_columns;

  for (auto [name, column] : _columns) {
    new_columns[name] = column->concat(other.getColumn(name));
  }

  _columns.clear();
  _num_rows = 0;
  other._columns.clear();
  other._num_rows = 0;

  return ColumnMap(std::move(new_columns));
}

ColumnMap ColumnMap::createStringColumnMapFromFile(
    const dataset::DataSourcePtr& source, char delimiter) {
  auto header_string = source->nextLine();
  if (!header_string.has_value()) {
    throw std::invalid_argument("Source was found to be empty.");
  }
  auto header = dataset::parsers::CSV::parseLine(*header_string, delimiter);

  std::vector<std::vector<std::string>> columns(header.size());
  while (auto line_str = source->nextLine()) {
    auto line = dataset::parsers::CSV::parseLine(*line_str, delimiter);
    if (line.size() != header.size()) {
      std::stringstream s;
      for (const auto& substr : line) {
        if (substr != line[0]) {
          s << delimiter;
        }
        s << substr;
      }
      throw std::invalid_argument(
          "Received a row with a different number of entries than in the "
          "header. Expected " +
          std::to_string(header.size()) + " entries but received " +
          std::to_string(line.size()) + " entries. Line: " + s.str());
    }
    for (size_t i = 0; i < columns.size(); i++) {
      columns.at(i).emplace_back(line.at(i));
    }
  }

  std::unordered_map<std::string, ColumnPtr> column_map;
  for (size_t i = 0; i < columns.size(); i++) {
    column_map[std::string(header.at(i))] =
        ValueColumn<std::string>::make(std::move(columns.at(i)));
  }

  return ColumnMap(column_map);
}

}  // namespace thirdai::data