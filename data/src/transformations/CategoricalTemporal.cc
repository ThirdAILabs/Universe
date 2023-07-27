#include "CategoricalTemporal.h"
#include <data/src/columns/ArrayColumns.h>
#include <optional>

namespace thirdai::data {

CategoricalTemporal::CategoricalTemporal(
    std::string user_column, std::string item_column,
    std::string timestamp_column, std::string output_column,
    size_t track_last_n, bool should_update_history, bool include_current_row,
    int64_t time_lag)
    : _user_column(std::move(user_column)),
      _item_column(std::move(item_column)),
      _timestamp_column(std::move(timestamp_column)),
      _output_column(std::move(output_column)),
      _track_last_n(track_last_n),
      _should_update_history(should_update_history),
      _include_current_row(include_current_row),
      _time_lag(time_lag) {}

ColumnMap CategoricalTemporal::apply(ColumnMap columns, State& state) const {
  auto user_col = columns.getValueColumn<std::string>(_user_column);
  auto item_col = columns.getArrayColumn<uint32_t>(_item_column);
  auto timestamp_col = columns.getValueColumn<int64_t>(_timestamp_column);

  auto item_history_tracker = state.getItemHistoryTracker(
      _user_column, _item_column, _timestamp_column);

  std::vector<std::vector<uint32_t>> last_n_items(user_col->numRows());

  for (size_t i = 0; i < user_col->numRows(); i++) {
    const std::string& user_id = user_col->value(i);
    int64_t timestamp = timestamp_col->value(i);

    std::vector<uint32_t> user_last_n_items;

    if (_include_current_row) {
      for (uint32_t item : item_col->row(i)) {
        user_last_n_items.push_back(item);
        if (user_last_n_items.size() >= _track_last_n) {
          break;
        }
      }
    }

    auto& user_item_history = (*item_history_tracker)[user_id];

    size_t seen = 0;
    for (auto it = user_item_history.rbegin();
         it != user_item_history.rend() &&
         user_last_n_items.size() < _track_last_n;
         ++it) {
      if (it->timestamp <= (timestamp - _time_lag)) {
        user_last_n_items.push_back(it->item);
      }
      seen++;
    }

    if (_should_update_history) {
      size_t n_outdated = user_item_history.size() - seen;
      user_item_history.erase(user_item_history.begin(),
                              user_item_history.begin() + n_outdated);

      for (uint32_t item : item_col->row(i)) {
        user_item_history.push_back({item, timestamp});
      }
    }

    last_n_items[i] = std::move(user_last_n_items);
  }

  std::optional<uint32_t> dim;
  if (item_col->dimension()) {
    dim = item_col->dimension()->dim;
  }
  auto output = ArrayColumn<uint32_t>::make(std::move(last_n_items), dim);
  columns.setColumn(_output_column, output);

  return columns;
}

}  // namespace thirdai::data