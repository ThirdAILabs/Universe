#include "ColdStartDataLoader.h"

namespace thirdai::automl::cold_start {

ColdStartDataLoader::ColdStartDataLoader(const data::ColumnMap& column_map,
                                         std::string text_column_name,
                                         std::string label_column_name,
                                         uint32_t batch_size,
                                         char column_delimiter,
                                         std::optional<char> label_delimiter)
    : DataLoader(batch_size),
      _text_column(column_map.getStringColumn(text_column_name)),
      _label_column(column_map.getTokenArrayColumn(label_column_name)),
      _row_idx(0),
      _text_column_name(std::move(text_column_name)),
      _label_column_name(std::move(label_column_name)),
      _column_delimiter(column_delimiter),
      _label_delimiter(label_delimiter),
      _header(getHeader()) {}

std::optional<std::vector<std::string>> ColdStartDataLoader::nextBatch() {
  std::vector<std::string> rows;

  while (auto row = getConcatenatedColumns()) {
    rows.push_back(std::move(*row));

    if (rows.size() == _target_batch_size) {
      break;
    }
  }

  if (rows.empty()) {
    return std::nullopt;
  }

  return rows;
}

std::optional<std::string> ColdStartDataLoader::nextLine() {
  if (_header) {
    std::string header = std::move(_header.value());
    _header = std::nullopt;
    return header;
  }
  return getConcatenatedColumns();
}

std::optional<std::string> ColdStartDataLoader::getConcatenatedColumns() {
  if (_row_idx == _text_column->numRows()) {
    return std::nullopt;
  }

  std::string row = (*_text_column)[_row_idx];

  auto labels = (*_label_column)[_row_idx];

  if (labels.size() > 1 && !_label_delimiter) {
    throw std::invalid_argument(
        "Expected label delimiter if the are multiple labels per sample.");
  }

  row.push_back(_column_delimiter);

  for (uint32_t label : labels) {
    row.append(std::to_string(label));
    row.push_back(*_label_delimiter);
  }

  _row_idx++;

  return row;
}

}  // namespace thirdai::automl::cold_start