#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/data_pipeline/Column.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class ColumnMap {
 public:
  explicit ColumnMap(std::unordered_map<std::string, ColumnPtr> columns);

  uint64_t numRows() const { return _num_rows; }

  BoltDatasetPtr convertToDataset(const std::vector<std::string>& column_names,
                                  uint32_t batch_size);

  std::shared_ptr<SparseValueColumn> getSparseValueColumn(
      const std::string& name);

  std::shared_ptr<DenseValueColumn> getDenseValueColumn(
      const std::string& name);

  std::shared_ptr<SparseArrayColumn> getSparseArrayColumn(
      const std::string& name);

  std::shared_ptr<DenseArrayColumn> getDenseArrayColumn(
      const std::string& name);

  ColumnPtr getColumn(const std::string& name);

  void addColumn(const std::string& name, ColumnPtr column) {
    _columns[name] = std::move(column);
  }

 private:
  std::vector<ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names);

  std::unordered_map<std::string, ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::dataset