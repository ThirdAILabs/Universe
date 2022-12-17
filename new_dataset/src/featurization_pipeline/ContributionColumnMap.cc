#include "ColumnMap.h"
#include <new_dataset/src/featurization_pipeline/Column.h>
#include <exception>
#include <stdexcept>
#include <string>

namespace thirdai::data {

ContributionColumnMap::ContributionColumnMap(
    std::unordered_map<std::string, columns::ContibutionColumnBasePtr>
        contribuition_columns)
    : _contribuition_columns(std::move(contribuition_columns)) {
  if (_contribuition_columns.empty()) {
    throw std::invalid_argument(
        "Cannot construct ContributionColumnMap from empty set of "
        "contributioncolumns.");
  }

  std::optional<uint64_t> num_rows = std::nullopt;
  for (auto& [_, contribuition_column] : _contribuition_columns) {
    if (num_rows && contribuition_column->numRows() != num_rows.value()) {
      throw std::invalid_argument(
          "All contribution columns must have the same number of rows.");
    }
    num_rows = contribuition_column->numRows();
  }
  _num_rows = num_rows.value();
}

columns::ContibutionColumnBasePtr ContributionColumnMap::getContributionColumn(
    const std::string& name) {
  if (!_contribuition_columns.count(name)) {
    throw std::invalid_argument("Unable to find column with name '" + name +
                                "'.");
  }
  return _contribuition_columns.at(name);
}

void ContributionColumnMap::setColumn(
    const std::string& name, columns::ContibutionColumnBasePtr column) {
  if (column->numRows() != _contribuition_columns.begin()->second->numRows()) {
    throw std::invalid_argument(
        "Cannot insert a Column with a different number of rows into a "
        "ColumnMap.");
  }
  _contribuition_columns[name] = std::move(column);
}

}  // namespace thirdai::data