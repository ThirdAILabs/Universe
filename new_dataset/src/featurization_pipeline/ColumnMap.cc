#include "ColumnMap.h"
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

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

ContributionColumnMap ColumnMap::getContributions(
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<float>>& raw_gradients,
    const std::optional<std::vector<std::vector<uint32_t>>>& indices) {
  auto output_columns = selectColumns(column_names);
  uint32_t num_columns = output_columns.size();
  if (numRows() != raw_gradients.size()) {
    throw std::invalid_argument(
        "gradients size and number of rows doesn't match.");
  }
  std::vector<uint32_t> column_dims;
  column_dims.push_back(0);
  for (const auto& col : output_columns) {
    if (auto output_dimension = col->dimension()) {
      column_dims.push_back(output_dimension->dim + column_dims.back());
    } else {
      throw std::invalid_argument(
          "Cannot convert column without dimension to get contributions.");
    }
  }
  std::vector<std::vector<float>> gradients =
      bolt::getPercentagesFromGradients(raw_gradients);
  std::vector<columns::CppTokenContributionColumn> contribution_columns(
      num_columns);
  for (uint32_t i = 0; i < num_columns; i++) {
    contribution_columns[i].resize(numRows());
  }
#pragma omp parallel for default(none) \
    shared(num_columns, indices, column_dims, gradients, contribution_columns)
  for (uint32_t vec_idx = 0; vec_idx < numRows(); vec_idx++) {
    std::vector<std::vector<columns::Contribution<uint32_t>>>
        contribuition_rows(num_columns);
    uint32_t start_index = 0;
    for (uint32_t i = 0; i < num_columns; i++) {
      if (indices) {
        uint32_t j;
        for (j = start_index; j < indices->at(vec_idx).size() &&
                              indices->at(vec_idx)[j] < column_dims[i + 1];
             j++) {
          contribuition_rows[i].push_back(columns::Contribution<uint32_t>(
              indices->at(vec_idx)[j], gradients[vec_idx][j]));
        }
        start_index = j;
      } else {
        uint32_t j;
        for (j = start_index; j < gradients[vec_idx].size(); j++) {
          if (j < column_dims[i + 1]) {
            contribuition_rows[i].push_back(
                columns::Contribution<uint32_t>(j, gradients[vec_idx][j]));
          } else {
            break;
          }
        }
        start_index = j;
      }
      contribution_columns[i].insert(contribuition_rows[i], vec_idx);
    }
  }
  std::unordered_map<std::string, columns::ContibutionColumnBasePtr>
      contribution_map;
  for (uint32_t i = 0; i < contribution_columns.size(); i++) {
    contribution_map[column_names[i]] =
        std::make_shared<columns::CppTokenContributionColumn>(
            contribution_columns[i]);
  }
  return ContributionColumnMap(contribution_map);
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
          dataset::SegmentedDenseFeatureVector vector(
              /* store_segment_feature_map= */ false);
          for (uint32_t i = 0; i < output_columns.size(); i++) {
            auto column = output_columns[i];
            vector.addFeatureSegment(column_dims[i]);
            column->appendRowToVector(vector, row_idx);
          }
          batch[vec_idx] = vector.toBoltVector();
        } else {
          dataset::SegmentedSparseFeatureVector vector(
              /* store_segment_feature_map= */ false);
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

void ColumnMap::setColumn(const std::string& name, columns::ColumnPtr column) {
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