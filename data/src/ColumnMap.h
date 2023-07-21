#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <data/src/columns/Column.h>
#include <dataset/src/Datasets.h>
#include <utils/Random.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::data {

class ColumnMap {
 public:
  explicit ColumnMap(std::unordered_map<std::string, ColumnPtr> columns);

  static ColumnMap fromMapInput(const automl::MapInput& sample);

  static ColumnMap fromMapInputBatch(const automl::MapInputBatch& samples);

  size_t numRows() const { return _num_rows; }

  template <typename T>
  ArrayColumnBasePtr<T> getArrayColumn(const std::string& name) const;

  template <typename T>
  ValueColumnBasePtr<T> getValueColumn(const std::string& name) const;

  ColumnPtr getColumn(const std::string& name) const;

  // Inserts a new column into the ColumnMap. If a column with the supplied name
  // already exists in the ColumnMap it will be overwritten.
  void setColumn(const std::string& name, ColumnPtr column);

  std::vector<std::string> columns() const;

  auto begin() const { return _columns.begin(); }

  auto end() const { return _columns.end(); }

  void shuffle(uint32_t seed = global_random::nextSeed());

  ColumnMap concat(ColumnMap& other);

  std::pair<ColumnMap, ColumnMap> split(size_t offset);

  void clear();

  static ColumnMap createStringColumnMapFromFile(
      const dataset::DataSourcePtr& source, char delimiter);

 private:
  std::unordered_map<std::string, ColumnPtr> _columns;
  size_t _num_rows;
};

}  // namespace thirdai::data