#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class ColumnMap {
 public:
  explicit ColumnMap(std::unordered_map<std::string, ColumnPtr> columns)
      : _columns(std::move(columns)) {
    std::optional<uint64_t> num_rows = std::nullopt;
    for (auto& [_, column] : _columns) {
      if (num_rows && column->numRows() != num_rows.value()) {
        throw std::invalid_argument(
            "All columns must have the same number of rows.");
      }
      num_rows = column->numRows();
    }
    if (!num_rows) {
      throw std::invalid_argument(
          "Cannot construct ColumnMap from empty set of columns.");
    }
    _num_rows = num_rows.value();
  }

  uint64_t numRows() const { return _num_rows; }

  std::vector<ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names) {
    std::vector<ColumnPtr> output_columns;
    output_columns.reserve(column_names.size());

    for (const auto& name : column_names) {
      if (!_columns.count(name)) {
        throw std::invalid_argument("'" + name +
                                    "' is not a valid column name.");
      }
      output_columns.push_back(_columns[name]);
    }

    return output_columns;
  }

 private:
  std::unordered_map<std::string, ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::dataset