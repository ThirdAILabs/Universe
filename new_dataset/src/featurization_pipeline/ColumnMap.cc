#include "ColumnMap.h"
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <exception>
#include <stdexcept>

namespace thirdai::data {

ColumnMap::ColumnMap(
    std::unordered_map<std::string, columns::ColumnPtr> columns)
    : _columns(std::move(columns)) {
  if (_columns.empty()) {
    throw std::invalid_argument(
        "Cannot construct ColumnMap from empty set of columns.");
  }

  std::optional<uint64_t> num_rows = std::nullopt;
  for (auto& [_, column] : _columns) {
    if (num_rows && column->numRows() != num_rows.value()) {
      throw std::invalid_argument(
          "All columns must have the same number of rows.");
    }
    num_rows = column->numRows();
  }
  _num_rows = num_rows.value();
}

dataset::BoltDatasetPtr ColumnMap::convertToDataset(
    const std::vector<std::string>& column_names, uint32_t batch_size) const {
  auto output_columns = selectColumns(column_names);

  std::vector<BoltBatch> output_batches;
  uint64_t num_batches = (numRows() + batch_size - 1) / batch_size;

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

  // TODO(Nicholas/Josh): Refactor to use new dataset without batches.
  for (uint64_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
    uint64_t curr_batch_size =
        std::min<uint64_t>(batch_size, numRows() - batch_idx * batch_size);

    std::vector<BoltVector> batch(curr_batch_size);

    std::exception_ptr exception = nullptr;

#pragma omp parallel for default(none)                             \
    shared(batch, curr_batch_size, all_cols_dense, output_columns, \
           column_dims, batch_idx, batch_size, exception)
    for (uint64_t vec_idx = 0; vec_idx < curr_batch_size; vec_idx++) {
      uint64_t row_idx = batch_idx * batch_size + vec_idx;

      try {
        if (all_cols_dense) {
          // TODO(Nicholas/Geordie): Refactor this into a unified row builder
          // class.
          dataset::SegmentedDenseFeatureVector vector;
          for (uint32_t i = 0; i < output_columns.size(); i++) {
            auto column = output_columns[i];
            vector.addFeatureSegment(column_dims[i]);
            column->appendRowToVector(vector, row_idx);
          }
          batch[vec_idx] = vector.toBoltVector();
        } else {
          dataset::SegmentedSparseFeatureVector vector;
          for (uint32_t i = 0; i < output_columns.size(); i++) {
            auto column = output_columns[i];
            vector.addFeatureSegment(column_dims[i]);
            column->appendRowToVector(vector, row_idx);
          }
          batch[vec_idx] = vector.toBoltVector();
        }
      } catch (std::exception& e) {
#pragma omp critical
        exception = std::current_exception();
      }
    }

    if (exception) {
      std::rethrow_exception(exception);
    }

    output_batches.emplace_back(std::move(batch));
  }

  return std::make_shared<dataset::BoltDataset>(std::move(output_batches));
}

std::vector<columns::ColumnPtr> ColumnMap::selectColumns(
    const std::vector<std::string>& column_names) const {
  std::vector<columns::ColumnPtr> output_columns;
  output_columns.reserve(column_names.size());

  for (const auto& name : column_names) {
    output_columns.push_back(getColumn(name));
  }

  return output_columns;
}

columns::TokenColumnPtr ColumnMap::getTokenColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::TokenColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to SparseValueColumn.");
  }
  return column;
}

columns::DenseFeatureColumnPtr ColumnMap::getDenseFeatureColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::DenseFeatureColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to DenseValueColumn.");
  }
  return column;
}

columns::SparseFeatureColumnPtr ColumnMap::getSparseFeatureColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::SparseFeatureColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to IndexValueColumn.");
  }
  return column;
}

columns::StringColumnPtr ColumnMap::getStringColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::StringColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to StringColumn.");
  }
  return column;
}

columns::TokenArrayColumnPtr ColumnMap::getTokenArrayColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::TokenArrayColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to SparseArrayColumn.");
  }
  return column;
}

columns::DenseArrayColumnPtr ColumnMap::getDenseArrayColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::DenseArrayColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to DenseArrayColumn.");
  }
  return column;
}

columns::SparseArrayColumnPtr ColumnMap::getSparseArrayColumn(
    const std::string& name) const {
  auto column =
      std::dynamic_pointer_cast<columns::SparseArrayColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument(
        "Column '" + name + "' cannot be converted to IndexValueArrayColumn.");
  }
  return column;
}

columns::ColumnPtr ColumnMap::getColumn(const std::string& name) const {
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

}  // namespace thirdai::data