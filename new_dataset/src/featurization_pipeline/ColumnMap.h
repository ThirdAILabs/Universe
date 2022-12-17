#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <_types/_uint32_t.h>
#include <dataset/src/Datasets.h>
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

class ContributionColumnMap {
 public:
  explicit ContributionColumnMap(
      std::unordered_map<std::string, columns::ContibutionColumnBasePtr>
          columns);

  columns::ContibutionColumnBasePtr getContributionColumn(
      const std::string& name);

  void setColumn(const std::string& name,
                 columns::ContibutionColumnBasePtr column);

 private:
  std::unordered_map<std::string, columns::ContibutionColumnBasePtr>
      _contribuition_columns;
  uint64_t _num_rows;
};

class ColumnMap {
 public:
  explicit ColumnMap(
      std::unordered_map<std::string, columns::ColumnPtr> columns);

  uint64_t numRows() const { return _num_rows; }

  // Converts each row in the dataset to a bolt vector by concatenating the
  // values of the specified columns in the order they were specified.
  dataset::BoltDatasetPtr convertToDataset(
      const std::vector<std::string>& column_names, uint32_t batch_size) const;

  ContributionColumnMap getContributions(
      const std::vector<std::string>& column_names,
      const std::vector<std::vector<float>>& gradients,
      const std::optional<std::vector<std::vector<uint32_t>>>& indices);

  // These methods get the column for the given name and use a dynamic cast to
  // convert it to the desired type. They will throw if the name does not match
  // any column or if the column does not have the specified type.
  columns::TokenColumnPtr getTokenColumn(const std::string& name) const;

  columns::DenseFeatureColumnPtr getDenseFeatureColumn(
      const std::string& name) const;

  columns::SparseFeatureColumnPtr getSparseFeatureColumn(
      const std::string& name) const;

  columns::StringColumnPtr getStringColumn(const std::string& name) const;

  columns::TokenArrayColumnPtr getTokenArrayColumn(
      const std::string& name) const;

  columns::DenseArrayColumnPtr getDenseArrayColumn(
      const std::string& name) const;

  columns::SparseArrayColumnPtr getSparseArrayColumn(
      const std::string& name) const;

  columns::ColumnPtr getColumn(const std::string& name) const;

  // Inserts a new column into the ColumnMap. If a column with the supplied name
  // already exists in the ColumnMap it will be overwritten.
  void setColumn(const std::string& name, columns::ColumnPtr column);

  std::vector<std::string> columns() const;

 private:
  std::vector<columns::ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names) const;

  std::unordered_map<std::string, columns::ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::data