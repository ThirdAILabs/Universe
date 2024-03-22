#include "ColumnMap.h"
#include <data/src/columns/Column.h>
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
  std::optional<size_t> num_rows = std::nullopt;
  for (auto& [_, column] : _columns) {
    if (num_rows && column->numRows() != num_rows.value()) {
      throw std::invalid_argument(
          "All columns must have the same number of rows.");
    }
    num_rows = column->numRows();
  }
  _num_rows = num_rows.value_or(0);
}

ColumnMap ColumnMap::fromMapInput(const automl::MapInput& sample) {
  std::unordered_map<std::string, ColumnPtr> columns;
  for (const auto& [name, row] : sample) {
    columns[name] = ValueColumn<std::string>::make({row});
  }

  return ColumnMap(std::move(columns));
}

ColumnMap ColumnMap::fromMapInputBatch(const automl::MapInputBatch& samples) {
  std::unordered_map<std::string, std::vector<std::string>> columns;

  for (const auto& sample : samples) {
    for (const auto& [name, row] : sample) {
      columns[name].push_back(row);
    }
  }

  std::unordered_map<std::string, ColumnPtr> column_map;
  for (auto& [name, samples] : columns) {
    column_map[name] = ValueColumn<std::string>::make(std::move(samples));
  }

  return ColumnMap(std::move(column_map));
}

template <typename T>
ArrayColumnBasePtr<T> ColumnMap::getArrayColumn(const std::string& name) const {
  auto column = ArrayColumnBase<T>::cast(getColumn(name));
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
  auto column = ValueColumnBase<T>::cast(getColumn(name));
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
  if (!containsColumn(name)) {
    throw std::invalid_argument("Unable to find column with name '" + name +
                                "'. ColumnMap contains columns " +
                                formatColumnNames() + ".");
  }
  return _columns.at(name);
}

bool ColumnMap::containsColumn(const std::string& name) const {
  return _columns.count(name);
}

void ColumnMap::setColumn(const std::string& name, ColumnPtr column) {
  if (column->numRows() != _num_rows) {
    throw std::invalid_argument(
        "Cannot insert a Column with a different number of rows into a "
        "ColumnMap.");
  }
  _columns[name] = std::move(column);
}

void ColumnMap::dropColumn(const std::string& name) {
  if (!containsColumn(name)) {
    throw std::runtime_error("Cannot drop column '" + name +
                             "' from column map with columns " +
                             formatColumnNames() + ".");
  }
  _columns.erase(name);
}

std::vector<std::string> ColumnMap::columns() const {
  std::vector<std::string> columns;
  for (auto const& map_entry : _columns) {
    columns.push_back(map_entry.first);
  }
  return columns;
}

ColumnMap ColumnMap::selectColumns(
    const std::vector<std::string>& columns) const {
  std::unordered_map<std::string, ColumnPtr> new_columns;
  for (const auto& name : columns) {
    new_columns[name] = getColumn(name);
  }
  return ColumnMap(std::move(new_columns));
}

void ColumnMap::shuffle(uint32_t seed) {
  std::vector<size_t> permutation(numRows());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(), std::mt19937{seed});

  for (auto& [_, column] : _columns) {
    column->shuffle(permutation);
  }
}

ColumnMap ColumnMap::permute(const std::vector<size_t>& permutation) const {
  std::unordered_map<std::string, ColumnPtr> new_columns;
  for (auto [name, column] : _columns) {
    new_columns[name] = column->permute(permutation);
  }
  return ColumnMap(std::move(new_columns));
}

ColumnMap ColumnMap::concat(ColumnMap& other) {
  if (this == &other) {
    throw std::invalid_argument("Cannot concatenate a ColumnMap with itself.");
  }

  if (!containsSameColumns(other)) {
    throw std::invalid_argument(
        "Cannot call concat on ColumnMaps with different columns. One "
        "ColumnMap has columns " +
        formatColumnNames() + " and the other has columns " +
        other.formatColumnNames() + ".");
  }

  std::unordered_map<std::string, ColumnPtr> new_columns;

  for (auto& [name, column] : _columns) {
    if (column->dim() != other.getColumn(name)->dim()) {
      throw std::invalid_argument(
          "Cannot concatenate column '" + name +
          "'. The dimensions don't match between column maps.");
    }
    new_columns[name] = column->concat(other.getColumn(name));
  }

  clear();
  other.clear();

  return ColumnMap(std::move(new_columns));
}

std::pair<ColumnMap, ColumnMap> ColumnMap::split(size_t starting_offset) {
  if (starting_offset >= numRows()) {
    throw std::invalid_argument(
        "invalid split offset " + std::to_string(starting_offset) +
        " for ColumnMap with " + std::to_string(numRows()) + " rows.");
  }

  std::unordered_map<std::string, ColumnPtr> front_columns;
  std::unordered_map<std::string, ColumnPtr> back_columns;

  for (auto& [name, column] : _columns) {
    auto [front, back] = column->split(starting_offset);
    front_columns[name] = front;
    back_columns[name] = back;
  }

  clear();

  return {ColumnMap(std::move(front_columns)),
          ColumnMap(std::move(back_columns))};
}

void ColumnMap::clear() {
  _columns.clear();
  _num_rows = 0;
}

bool ColumnMap::containsSameColumns(const ColumnMap& other) const {
  if (_columns.size() != other._columns.size()) {
    return false;
  }

  return std::all_of(
      _columns.begin(), _columns.end(),
      [&other](const auto& col) { return other.containsColumn(col.first); });
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

std::string ColumnMap::formatColumnNames() const {
  std::string column_names = "[";
  for (const auto& [name, _] : _columns) {
    column_names += "'" + name + "', ";
  }
  if (column_names.size() > 2) {
    column_names.pop_back();  // remove last space
    column_names.pop_back();  // remove last comma
  }

  column_names += "]";

  return column_names;
}

}  // namespace thirdai::data