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

  // Converts each row in the dataset to a bolt vector by concatenating the
  // values of the specified columns in the order they were specified.
  BoltDatasetPtr convertToDataset(const std::vector<std::string>& column_names,
                                  uint32_t batch_size) const;

  // These methods get the column for the given name and use a dynamic cast to
  // convert it to the desired type. They will throw if the name does not match
  // any column or if the column does not have the specified type.
  std::shared_ptr<SparseValueColumn> getSparseValueColumn(
      const std::string& name) const;

  std::shared_ptr<DenseValueColumn> getDenseValueColumn(
      const std::string& name) const;

  std::shared_ptr<IndexValueColumn> getIndexValueColumn(
      const std::string& name) const;

  std::shared_ptr<SparseArrayColumn> getSparseArrayColumn(
      const std::string& name) const;

  std::shared_ptr<DenseArrayColumn> getDenseArrayColumn(
      const std::string& name) const;

  std::shared_ptr<IndexValueArrayColumn> getIndexValueArrayColumn(
      const std::string& name) const;

  ColumnPtr getColumn(const std::string& name) const;

  void setColumn(const std::string& name, ColumnPtr column) {
    _columns[name] = std::move(column);
  }

 private:
  std::vector<ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names) const;

  std::unordered_map<std::string, ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::dataset