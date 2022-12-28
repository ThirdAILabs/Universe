#include "ColdStartDataLoader.h"
#include <stdexcept>

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

  while (auto row = nextLine()) {
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
  return getNextRowAsString();
}

std::optional<std::string> ColdStartDataLoader::getNextRowAsString() {
  if (_row_idx == _text_column->numRows()) {
    return std::nullopt;
  }

  std::string row = getLabelsAsString();

  row.push_back(_column_delimiter);

  row.append((*_text_column)[_row_idx]);

  _row_idx++;

  return row;
}

std::string ColdStartDataLoader::getLabelsAsString() {
  auto labels = (*_label_column)[_row_idx];

  if (labels.size() == 0) {
    throw std::invalid_argument("Expected at least 1 label per row.");
  }
  if (labels.size() > 1 && !_label_delimiter) {
    throw std::invalid_argument(
        "Expected label delimiter if the are multiple labels per sample.");
  }

  std::string labels_str = std::to_string(labels[0]);

  for (uint32_t label_idx = 1; label_idx < labels.size(); label_idx++) {
    labels_str.push_back(*_label_delimiter);
    labels_str.append(std::to_string(labels[label_idx]));
  }

  return labels_str;
}

}  // namespace thirdai::automl::cold_start