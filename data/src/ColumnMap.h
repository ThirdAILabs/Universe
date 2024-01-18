#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <data/src/columns/Column.h>
#include <dataset/src/Datasets.h>
#include <utils/Random.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

  bool containsColumn(const std::string& name) const;

  // Inserts a new column into the ColumnMap. If a column with the supplied name
  // already exists in the ColumnMap it will be overwritten.
  void setColumn(const std::string& name, ColumnPtr column);

  void dropColumn(const std::string& name);

  std::vector<std::string> columns() const;

  auto begin() const { return _columns.begin(); }

  auto end() const { return _columns.end(); }

  /**
   * Shuffles the ColumnMap in place.
   */
  void shuffle(uint32_t seed = global_random::nextSeed());

  /**
   * Creates a new column map whose rows are permuted in the given order
   */
  ColumnMap permute(const std::vector<size_t>& permutation) const;

  /**
   * Concatenates with another ColumnMap, returning the result. This will
   * consume both ColumnMaps so that values can be moved without copying when
   * possible.
   */
  ColumnMap concat(ColumnMap& other);

  /**
   * Splits the ColumnMap in two, returning two new ColumnMaps. Consumes the
   * ColumnMap it is called on so that values can be moved without copying when
   * possible. The first ColumnMap will have rows [0, starting_offset), and the
   * second will have rows [starting_offset, num_rows).
   */
  std::pair<ColumnMap, ColumnMap> split(size_t starting_offset);

  static ColumnMap createStringColumnMapFromFile(
      const dataset::DataSourcePtr& source, char delimiter);

  std::string debugStr() const;

 private:
  void clear();

  bool containsSameColumns(const ColumnMap& other) const;

  std::string formatColumnNames() const;

  std::unordered_map<std::string, ColumnPtr> _columns;
  size_t _num_rows;
};

}  // namespace thirdai::data