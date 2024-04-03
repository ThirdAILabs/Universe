#pragma once

#include <data/src/ColumnMap.h>
#include <dataset/src/DataSource.h>

namespace thirdai::data {

using dataset::DataSourcePtr;

class ColumnMapIterator {
 public:
  // NWP Task requires less rows per load
  static constexpr size_t DEFAULT_ROWS_PER_LOAD = 100;

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

}  // namespace thirdai::data