#pragma once

#include <data/src/ColumnMap.h>
#include <dataset/src/DataSource.h>

namespace thirdai::data {

using dataset::DataSourcePtr;

class ColumnMapIterator {
 public:
  ColumnMapIterator(DataSourcePtr data_source, char delimiter,
                    size_t rows_per_load);

  std::optional<ColumnMap> next();

  ColumnMap emptyColumnMap() const;

 private:
  ColumnMap makeColumnMap(
      std::vector<std::vector<std::string>>&& columns) const;

  DataSourcePtr _data_source;
  char _delimiter;
  size_t _rows_per_load;

  std::vector<std::string> _column_names;
};

}  // namespace thirdai::data