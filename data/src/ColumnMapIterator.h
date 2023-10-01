#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/State.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/DataSource.h>
#include <memory>

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

 private:
  DataSourcePtr _data_source;
  size_t _rows_per_load;

  std::vector<std::string> _column_names;
};

class TransformIterator final : public ColumnMapIterator {
 public:
  TransformIterator(ColumnMapIteratorPtr iter, TransformationPtr transform,
                    StatePtr state)
      : _iter(std::move(iter)),
        _transform(std::move(transform)),
        _state(std::move(state)) {}

  static auto make(ColumnMapIteratorPtr iter, TransformationPtr transform,
                   StatePtr state) {
    return std::make_shared<TransformIterator>(
        std::move(iter), std::move(transform), std::move(state));
  }

  std::optional<ColumnMap> next() final {
    auto next_column_map = _iter->next();
    if (next_column_map) {
      next_column_map = _transform->apply(std::move(*next_column_map), *_state);
    }
    return next_column_map;
  }

  void restart() final { _iter->restart(); }

  std::string resourceName() const final { return _iter->resourceName(); }

 private:
  ColumnMapIteratorPtr _iter;
  TransformationPtr _transform;
  StatePtr _state;
};

}  // namespace thirdai::data