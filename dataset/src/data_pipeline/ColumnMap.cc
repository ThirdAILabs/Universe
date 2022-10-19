#include "ColumnMap.h"

namespace thirdai::dataset {

ColumnMap::ColumnMap(std::unordered_map<std::string, ColumnPtr> columns)
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

BoltDatasetPtr ColumnMap::convertToDataset(
    const std::vector<std::string>& column_names, uint32_t batch_size) {
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

#pragma omp parallel for default(none)                             \
    shared(batch, curr_batch_size, all_cols_dense, output_columns, \
           column_dims, batch_idx, batch_size)
    for (uint64_t vec_idx = 0; vec_idx < curr_batch_size; vec_idx++) {
      uint64_t row_idx = batch_idx * batch_size + vec_idx;

      if (all_cols_dense) {
        // TODO(Nicholas/Geordie): Refactor this into a unified row builder
        // class.
        SegmentedDenseFeatureVector vector;
        for (uint32_t i = 0; i < output_columns.size(); i++) {
          auto column = output_columns[i];
          vector.addFeatureSegment(column_dims[i]);
          column->appendRowToVector(vector, row_idx);
        }
        batch[vec_idx] = vector.toBoltVector();
      } else {
        SegmentedSparseFeatureVector vector;
        for (uint32_t i = 0; i < output_columns.size(); i++) {
          auto column = output_columns[i];
          vector.addFeatureSegment(column_dims[i]);
          column->appendRowToVector(vector, row_idx);
        }
        batch[vec_idx] = vector.toBoltVector();
      }
    }

    output_batches.emplace_back(std::move(batch));
  }

  return std::make_shared<BoltDataset>(std::move(output_batches));
}

std::vector<ColumnPtr> ColumnMap::selectColumns(
    const std::vector<std::string>& column_names) {
  std::vector<ColumnPtr> output_columns;
  output_columns.reserve(column_names.size());

  for (const auto& name : column_names) {
    output_columns.push_back(getColumn(name));
  }

  return output_columns;
}

std::shared_ptr<IntegerValueColumn> ColumnMap::getIntegerValueColumn(
    const std::string& name) {
  auto column = std::dynamic_pointer_cast<IntegerValueColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to IntegerValueColumn.");
  }
  return column;
}

std::shared_ptr<FloatValueColumn> ColumnMap::getFloatValueColumn(
    const std::string& name) {
  auto column = std::dynamic_pointer_cast<FloatValueColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to FloatValueColumn.");
  }
  return column;
}

std::shared_ptr<IntegerArrayColumn> ColumnMap::getIntegerArrayColumn(
    const std::string& name) {
  auto column = std::dynamic_pointer_cast<IntegerArrayColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to IntegerArrayColumn.");
  }
  return column;
}

std::shared_ptr<FloatArrayColumn> ColumnMap::getFloatArrayColumn(
    const std::string& name) {
  auto column = std::dynamic_pointer_cast<FloatArrayColumn>(getColumn(name));
  if (!column) {
    throw std::invalid_argument("Column '" + name +
                                "' cannot be converted to FloatArrayColumn.");
  }
  return column;
}

ColumnPtr ColumnMap::getColumn(const std::string& name) {
  if (!_columns.count(name)) {
    throw std::invalid_argument("Unable to find column with name '" + name +
                                "'.");
  }
  return _columns.at(name);
}

}  // namespace thirdai::dataset