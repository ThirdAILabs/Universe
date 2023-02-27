#include "ColdStartDataSource.h"
#include <stdexcept>

namespace thirdai::dataset {

ColdStartDataSource::ColdStartDataSource(const data::ColumnMap& column_map,
                                         std::string text_column_name,
                                         std::string label_column_name,
                                         char column_delimiter,
                                         std::optional<char> label_delimiter,
                                         std::string resource_name)
    : _text_column(column_map.getStringColumn(text_column_name)),
      _label_column(column_map.getStringColumn(label_column_name)),
      _row_idx(0),
      _text_column_name(std::move(text_column_name)),
      _label_column_name(std::move(label_column_name)),
      _column_delimiter(column_delimiter),
      _label_delimiter(label_delimiter),
      _header(getHeader()),
      _resource_name(std::move(resource_name)) {}

std::optional<std::vector<std::string>> ColdStartDataSource::nextBatch(
    size_t target_batch_size) {
  std::vector<std::string> rows;

  while (auto row = nextLine()) {
    rows.push_back(std::move(*row));

    if (rows.size() == target_batch_size) {
      break;
    }
  }

  if (rows.empty()) {
    return std::nullopt;
  }

  return rows;
}

std::optional<std::string> ColdStartDataSource::nextLine() {
  if (_header) {
    std::string header = std::move(_header.value());
    _header = std::nullopt;
    return header;
  }
  return getNextRowAsString();
}

std::optional<std::string> ColdStartDataSource::getNextRowAsString() {
  if (_row_idx == _text_column->numRows()) {
    return std::nullopt;
  }

  std::string row = (*_label_column)[_row_idx];

  row.push_back(_column_delimiter);

  row.append((*_text_column)[_row_idx]);

  _row_idx++;

  return row;
}

}  // namespace thirdai::dataset