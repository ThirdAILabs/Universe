#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/data_pipeline/Column.h>
#include <memory>
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

  std::vector<BoltVector> convertToDataset(
      const std::vector<std::string>& column_names) {
    auto output_columns = selectColumns(column_names);

    std::vector<BoltVector> output_vectors;
    output_vectors.reserve(numRows());

    bool all_cols_dense = true;
    std::vector<uint32_t> column_dims;
    for (const auto& col : output_columns) {
      if (auto output_dimension = col->dimension()) {
        all_cols_dense = all_cols_dense && output_dimension->is_dense;
        column_dims.push_back(output_dimension->dim);
      } else {
        throw std::invalid_argument(
            "Cannot convert column without dimension to dataset");
      }
    }

    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      if (all_cols_dense) {
        // TODO(Nicholas/Geordie): Refactor this into a unified row builder
        // class.
        SegmentedDenseFeatureVector vector;
        for (uint32_t i = 0; i < output_columns.size(); i++) {
          auto column = output_columns[i];
          vector.addFeatureSegment(column_dims[i]);
          column->appendRowToVector(vector, row_index);
        }
        output_vectors.push_back(vector.toBoltVector());
      } else {
        SegmentedSparseFeatureVector vector;
        for (uint32_t i = 0; i < output_columns.size(); i++) {
          auto column = output_columns[i];
          vector.addFeatureSegment(column_dims[i]);
          column->appendRowToVector(vector, row_index);
        }
        output_vectors.push_back(vector.toBoltVector());
      }
    }

    return output_vectors;
  }

  std::shared_ptr<IntegerValueColumn> getIntegerValueColumn(
      const std::string& name) {
    auto column =
        std::dynamic_pointer_cast<IntegerValueColumn>(getColumn(name));
    if (!column) {
      throw std::invalid_argument(
          "Column '" + name + "' cannot be converted to IntegerValueColumn.");
    }
    return column;
  }

  std::shared_ptr<FloatValueColumn> getFloatValueColumn(
      const std::string& name) {
    auto column = std::dynamic_pointer_cast<FloatValueColumn>(getColumn(name));
    if (!column) {
      throw std::invalid_argument("Column '" + name +
                                  "' cannot be converted to FloatValueColumn.");
    }
    return column;
  }

  std::shared_ptr<IntegerArrayColumn> getIntegerArrayColumn(
      const std::string& name) {
    auto column =
        std::dynamic_pointer_cast<IntegerArrayColumn>(getColumn(name));
    if (!column) {
      throw std::invalid_argument(
          "Column '" + name + "' cannot be converted to IntegerArrayColumn.");
    }
    return column;
  }

  std::shared_ptr<FloatArrayColumn> getFloatArrayColumn(
      const std::string& name) {
    auto column = std::dynamic_pointer_cast<FloatArrayColumn>(getColumn(name));
    if (!column) {
      throw std::invalid_argument("Column '" + name +
                                  "' cannot be converted to FloatArrayColumn.");
    }
    return column;
  }

  ColumnPtr getColumn(const std::string& name) {
    if (!_columns.count(name)) {
      throw std::invalid_argument("Unable to find column with name '" + name +
                                  "'.");
    }
    return _columns.at(name);
  }

  void addColumn(const std::string& name, ColumnPtr column) {
    _columns[name] = std::move(column);
  }

 private:
  std::vector<ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names) {
    std::vector<ColumnPtr> output_columns;
    output_columns.reserve(column_names.size());

    for (const auto& name : column_names) {
      output_columns.push_back(getColumn(name));
    }

    return output_columns;
  }

  std::unordered_map<std::string, ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::dataset