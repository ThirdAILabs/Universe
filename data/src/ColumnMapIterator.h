#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <nlohmann/json.hpp>

namespace thirdai::data {

using dataset::DataSourcePtr;

class ColumnMapIterator {
 public:
  static constexpr size_t DEFAULT_ROWS_PER_LOAD = 1000000;

  virtual std::optional<ColumnMap> next() = 0;

  virtual void restart() = 0;

  virtual std::string resourceName() const = 0;

  virtual ~ColumnMapIterator() = default;
};

using ColumnMapIteratorPtr = std::shared_ptr<ColumnMapIterator>;

class CsvIterator final : public ColumnMapIterator {
 public:
  CsvIterator(const std::string& filename, char delimiter,
              size_t rows_per_load = DEFAULT_ROWS_PER_LOAD);

  CsvIterator(DataSourcePtr data_source, char delimiter,
              size_t rows_per_load = DEFAULT_ROWS_PER_LOAD);

  static auto make(DataSourcePtr data_source, char delimiter,
                   size_t rows_per_load = DEFAULT_ROWS_PER_LOAD) {
    return std::make_shared<CsvIterator>(std::move(data_source), delimiter,
                                         rows_per_load);
  }

  static ColumnMap all(DataSourcePtr data_source, char delimiter);

  std::optional<ColumnMap> next() final;

  void restart() final;

  std::string resourceName() const final {
    return _data_source->resourceName();
  }

 private:
  DataSourcePtr _data_source;
  char _delimiter;
  size_t _rows_per_load;

  std::vector<std::string> _column_names;
};

class JsonIterator final : public ColumnMapIterator {

using json = nlohmann::json;
 public:
  JsonIterator(DataSourcePtr data_source, std::vector<std::string> column_names,
               size_t rows_per_load = DEFAULT_ROWS_PER_LOAD);

  static auto make(DataSourcePtr data_source,
                   std::vector<std::string> column_names,
                   size_t rows_per_load = DEFAULT_ROWS_PER_LOAD) {
    return std::make_shared<JsonIterator>(
        std::move(data_source), std::move(column_names), rows_per_load);
  }

  std::optional<ColumnMap> next() final;

  void restart() final;

  std::string resourceName() const final {
    return _data_source->resourceName();
  }

  template <typename T>
  void extractColumnData(const std::vector<std::string>& rows,
                         const std::string& column_name, std::vector<T>& vec);

  static void validateJsonRow(const json &row,
                                   const std::string& column_name);

 private:
  DataSourcePtr _data_source;
  size_t _rows_per_load;

  std::vector<std::string> _column_names;
};

class TransformedIterator final : public ColumnMapIterator {
 public:
  TransformedIterator(ColumnMapIteratorPtr iter,
                      TransformationPtr transformation, StatePtr state)
      : _iter(std::move(iter)),
        _transformation(std::move(transformation)),
        _state(std::move(state)) {}

  static auto make(ColumnMapIteratorPtr iter, TransformationPtr transformation,
                   StatePtr state) {
    return std::make_shared<TransformedIterator>(
        std::move(iter), std::move(transformation), std::move(state));
  }

  std::optional<ColumnMap> next() final {
    auto next_columns = _iter->next();
    if (next_columns) {
      next_columns = _transformation->apply(std::move(*next_columns), *_state);
    }
    return next_columns;
  }

  void restart() final { _iter->restart(); }

  std::string resourceName() const final { return _iter->resourceName(); }

 private:
  ColumnMapIteratorPtr _iter;
  TransformationPtr _transformation;
  StatePtr _state;
};

}  // namespace thirdai::data