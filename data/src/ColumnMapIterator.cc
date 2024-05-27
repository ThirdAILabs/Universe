#include "ColumnMapIterator.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/Column.h>
#include <data/src/columns/ValueColumns.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/utils/CsvParser.h>
#include <nlohmann/json.hpp>
#include <exception>
#include <limits>
#include <stdexcept>

namespace thirdai::data {

using json = nlohmann::json;

ColumnMap makeColumnMap(std::vector<std::vector<std::string>>&& columns,
                        const std::vector<std::string>& column_names) {
  std::unordered_map<std::string, ColumnPtr> column_map;

  for (size_t i = 0; i < columns.size(); i++) {
    column_map[column_names[i]] =
        ValueColumn<std::string>::make(std::move(columns[i]));
  }

  return ColumnMap(std::move(column_map));
}

CsvIterator::CsvIterator(const std::string& filename, char delimiter,
                         size_t rows_per_load)
    : CsvIterator(dataset::FileDataSource::make(filename), delimiter,
                  rows_per_load) {}

CsvIterator::CsvIterator(DataSourcePtr data_source, char delimiter,
                         size_t rows_per_load)
    : _data_source(
          dataset::CsvDataSource::make(std::move(data_source), delimiter)),
      _delimiter(delimiter),
      _rows_per_load(rows_per_load) {
  _data_source->restart();
  auto header = _data_source->nextLine();
  if (!header.has_value()) {
    throw std::invalid_argument("DataSource was found to be empty.");
  }
  _column_names = dataset::parsers::CSV::parseLine(*header, _delimiter);
}

ColumnMap CsvIterator::all(DataSourcePtr data_source, char delimiter) {
  CsvIterator data_iter(std::move(data_source), delimiter,
                        std::numeric_limits<size_t>::max());

  auto data = data_iter.next();

  if (!data) {
    throw std::invalid_argument("Unable to load data from '" +
                                data_iter.resourceName() + "'.");
  }

  return *data;
}

std::optional<ColumnMap> CsvIterator::next() {
  auto rows = _data_source->nextBatch(_rows_per_load);
  if (!rows) {
    return std::nullopt;
  }

  std::vector<std::vector<std::string>> columns(
      _column_names.size(), std::vector<std::string>(rows->size()));

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(rows, columns, error)
  for (size_t row_idx = 0; row_idx < rows->size(); row_idx++) {
    try {
      const auto& row = rows->at(row_idx);
      auto row_columns = dataset::parsers::CSV::parseLine(row, _delimiter);
      if (row_columns.size() != _column_names.size()) {
        throw std::invalid_argument(
            "Expected " + std::to_string(_column_names.size()) +
            " columns. But received row '" + row + "' with " +
            std::to_string(row_columns.size()) + " columns.");
      }

      for (size_t i = 0; i < columns.size(); i++) {
        columns[i][row_idx] = std::move(row_columns[i]);
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  return makeColumnMap(std::move(columns), _column_names);
}

void CsvIterator::restart() {
  _data_source->restart();
  _data_source->nextLine();  // To clear the header.
}

JsonIterator::JsonIterator(DataSourcePtr data_source,
                           std::vector<std::string> column_names,
                           size_t rows_per_load)
    : _data_source(std::move(data_source)),
      _rows_per_load(rows_per_load),
      _column_names(std::move(column_names)) {}

void validateJsonRow(const json& row, const std::string& column_name) {
  if (!row.is_object()) {
    throw std::invalid_argument("Expected row to be a JSON object");
  }

  if (!row.contains(column_name)) {
    throw std::invalid_argument("Expected row to contain key '" + column_name +
                                "'.");
  }
}

template <typename T>
void extractColumnData(const std::vector<std::string>& rows,
                       const std::string& column_name, std::vector<T>& vec) {
  std::exception_ptr error;

#pragma omp parallel for default(none) shared(rows, vec, column_name, error)
  for (size_t row_idx = 0; row_idx < rows.size(); row_idx++) {
    try {
      auto row = json::parse(rows.at(row_idx));
      validateJsonRow(row, column_name);
      auto& cell = row.at(column_name);
      vec[row_idx] = cell.get<T>();
    } catch (const std::exception& e) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }
}
std::optional<ColumnMap> JsonIterator::next() {
  auto rows = _data_source->nextBatch(_rows_per_load);
  if (!rows) {
    return std::nullopt;
  }

  std::unordered_map<std::string, ColumnPtr> column_map;

  auto first_row = json::parse(rows->at(0));
  for (const auto& column_name : _column_names) {
    // We check the data-type for each column here.

    validateJsonRow(first_row, column_name);

    auto first_cell = first_row[column_name];
    if (first_cell.is_number_integer()) {
      std::vector<uint32_t> all_ints(rows->size());
      extractColumnData(rows.value(), column_name, all_ints);
      column_map[column_name] =
          ValueColumn<uint32_t>::make(std::move(all_ints), std::nullopt);
    } else if (first_cell.is_array() && !first_cell.empty()) {
      if (first_cell[0].is_number_integer()) {
        std::vector<std::vector<uint32_t>> all_vec_ints(rows->size());
        extractColumnData(rows.value(), column_name, all_vec_ints);
        column_map[column_name] =
            ArrayColumn<uint32_t>::make(std::move(all_vec_ints));
      } else if (first_cell[0].is_string()) {
        std::vector<std::vector<std::string>> all_vec_strings(rows->size());
        extractColumnData(rows.value(), column_name, all_vec_strings);
        column_map[column_name] =
            ArrayColumn<std::string>::make(std::move(all_vec_strings));

      } else {
        throw std::invalid_argument(
            "Expected values of fields in row to be "
            "string/integer/List[integer]/List[string].");
      }
    } else if (first_cell.is_string()) {
      std::vector<std::string> all_strings(rows->size());
      extractColumnData(rows.value(), column_name, all_strings);
      column_map[column_name] =
          ValueColumn<std::string>::make(std::move(all_strings));
    } else {
      throw std::invalid_argument(
          "Expected values of fields in row to be "
          "string/integer/List[integer]/List[string].");
    }
  }

  return ColumnMap(std::move(column_map));
}

void JsonIterator::restart() { _data_source->restart(); }

}  // namespace thirdai::data