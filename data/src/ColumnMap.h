#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <data/src/columns/Column.h>
#include <dataset/src/Datasets.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

class ColumnMap {
 public:
  explicit ColumnMap(std::unordered_map<std::string, ColumnPtr> columns);

  uint64_t numRows() const { return _num_rows; }

  template <typename T>
  ArrayColumnPtr<T> getArrayColumn(const std::string& name) const;

  template <typename T>
  ValueColumnPtr<T> getValueColumn(const std::string& name) const;

  ColumnPtr getColumn(const std::string& name) const;

  // Inserts a new column into the ColumnMap. If a column with the supplied name
  // already exists in the ColumnMap it will be overwritten.
  void setColumn(const std::string& name, ColumnPtr column);

  std::vector<std::string> columns() const;

  static ColumnMap createStringColumnMapFromFile(
      const dataset::DataSourcePtr& source, char delimiter);

 private:
  std::vector<ColumnPtr> selectColumns(
      const std::vector<std::string>& column_names) const;

  std::unordered_map<std::string, ColumnPtr> _columns;
  uint64_t _num_rows;
};

}  // namespace thirdai::data