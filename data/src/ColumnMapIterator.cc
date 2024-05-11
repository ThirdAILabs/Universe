#include "ColumnMapIterator.h"
#include <data/src/columns/ValueColumns.h>
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/utils/CsvParser.h>
#include <nlohmann/json.hpp>
#include <exception>
#include <limits>
#include <stdexcept>

namespace thirdai::data {

using json = nlohmann::json;
using ColumnData = std::variant<std::string, std::vector<std::string>, uint32_t, std::vector<uint32_t>>;

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

std::optional<ColumnMap> JsonIterator::next() {
    auto rows = _data_source->nextBatch(_rows_per_load);
    if (!rows) {
        return std::nullopt;
    }

    std::vector<std::vector<ColumnData>> columns(_column_names.size(), std::vector<ColumnData>(rows->size()));
    std::exception_ptr error;
    std::vector<std::string> column_types(_column_names.size());

    #pragma omp parallel for default(none) shared(rows, columns, column_types, error)
    for (size_t row_idx = 0; row_idx < rows->size(); row_idx++) {
        try {
            auto row = json::parse(rows->at(row_idx));
            if (!row.is_object()) {
                throw std::invalid_argument("Expected row to be a json object but received '" + rows->at(row_idx) + "'.");
            }

            for (size_t i = 0; i < _column_names.size(); i++) {
                if (!row.contains(_column_names[i])) {
                    throw std::invalid_argument("Expected row to contain key '" + _column_names[i] + "'.");
                }

                auto& cell = row[_column_names[i]];
                if (cell.is_number_integer()) {
                    columns[i][row_idx] = cell.get<uint32_t>();
                    if(row_idx == 0){
                      column_types[i] = "int";
                    }
                } else if (cell.is_array() && !cell.empty()) {
                    if (cell[0].is_number_integer()) {
                        columns[i][row_idx] = cell.get<std::vector<uint32_t>>();
                        if(row_idx == 0){
                          column_types[i] = "vec-int";
                        }
                    } else {
                        columns[i][row_idx] = cell.get<std::vector<std::string>>();
                        if(row_idx == 0){
                          column_types[i] = "vec-string";
                        }
                    }
                } else if(cell.is_string()){
                    columns[i][row_idx] = cell.get<std::string>();
                    if(row_idx == 0){
                      column_types[i] = "string";
                    }
                }else{
                  throw std::invalid_argument(
                    "Expected values of fields in row to be string/integer/List[integer]/List[string].");
                }
            }
        } catch (...) {
            #pragma omp critical
            error = std::current_exception();
        }
    }

    if (error) {
        std::rethrow_exception(error);
    }


    std::unordered_map<std::string, ColumnPtr> column_map;
    for (size_t column_idx=0; column_idx<columns.size(); column_idx+=1) {
        if(column_types[column_idx] == "int"){
          std::vector<uint32_t> all_ints;
          for (auto & i : columns[column_idx]) {
            all_ints.push_back(std::get<uint32_t>(i));  
        }

        column_map[_column_names[column_idx]] =
            ValueColumn<uint32_t>::make(std::move(all_ints));
        }else if(column_types[column_idx] == "string"){
        std::vector<std::string> all_strings;
        for (auto & i : columns[column_idx]) {
            all_strings.push_back(std::get<std::string>(i));  
        }

        column_map[_column_names[column_idx]] =
            ValueColumn<std::string>::make(std::move(all_strings));
        }else if(column_types[column_idx] == "vec-int"){
          std::vector<std::vector<uint32_t>> all_vec_ints;
        for (auto & i : columns[column_idx]) {
            all_vec_ints.push_back(std::get<std::vector<uint32_t>>(i));  
        }

        column_map[_column_names[column_idx]] =
            ArrayColumn<uint32_t>::make(std::move(all_vec_ints));
        }else{
          std::vector<std::vector<std::string>> all_vec_strings;
          for (auto & i : columns[column_idx]) {
              all_vec_strings.push_back(std::get<std::vector<std::string>>(i));
          }

        column_map[_column_names[column_idx]] =
            ArrayColumn<std::string>::make(std::move(all_vec_strings));
        }
        
    }

    return ColumnMap(std::move(column_map));
}


void JsonIterator::restart() { _data_source->restart(); }

}  // namespace thirdai::data