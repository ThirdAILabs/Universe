#include "CategoricalTemporal.h"
#include <data/src/columns/ArrayColumns.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace thirdai::data {

CategoricalTemporal::CategoricalTemporal(
    std::string user_column, std::string item_column,
    std::string timestamp_column, std::string output_column,
    std::string tracker_key, size_t track_last_n, bool should_update_history,
    bool include_current_row, int64_t time_lag)
    : _user_column(std::move(user_column)),
      _item_column(std::move(item_column)),
      _timestamp_column(std::move(timestamp_column)),
      _output_column(std::move(output_column)),
      _tracker_key(std::move(tracker_key)),
      _track_last_n(track_last_n),
      _should_update_history(should_update_history),
      _include_current_row(include_current_row),
      _time_lag(time_lag) {}

ColumnMap CategoricalTemporal::apply(ColumnMap columns, State& state) const {
  auto user_col = columns.getValueColumn<std::string>(_user_column);
  auto item_col = columns.getArrayColumn<uint32_t>(_item_column);
  auto timestamp_col = columns.getValueColumn<int64_t>(_timestamp_column);

  auto& item_history_tracker = state.getItemHistoryTracker(_tracker_key);

  std::vector<std::vector<uint32_t>> last_n_items(user_col->numRows());

  for (size_t i = 0; i < user_col->numRows(); i++) {
    const std::string& user_id = user_col->value(i);
    int64_t timestamp = timestamp_col->value(i);

    if (timestamp < item_history_tracker.last_timestamp) {
      throw std::invalid_argument("Expected increasing timestamps in column '" +
                                  _timestamp_column + "'.");
    }
    item_history_tracker.last_timestamp = timestamp;

    std::vector<uint32_t> user_last_n_items;

    if (_include_current_row && _time_lag == 0) {
      auto row = item_col->row(i);
      // The item history is LIFO, so the last items added are the first popped.
      // We add the current row's items in reverse order here so that it is
      // consistent with the order they will be popped from the item history if
      // should_update_history is true.
      for (size_t i = 1; i <= row.size(); i++) {
        user_last_n_items.push_back(row[row.size() - i]);
        if (user_last_n_items.size() >= _track_last_n) {
          break;
        }
      }
    }

    auto& user_item_history = item_history_tracker.trackers[user_id];

    size_t seen = 0;
    for (auto it = user_item_history.rbegin();
         it != user_item_history.rend() &&
         user_last_n_items.size() < _track_last_n;
         i++ t) {
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

  auto output =
      ArrayColumn<uint32_t>::make(std::move(last_n_items), item_col->dim());
  columns.setColumn(_output_column, output);

  return columns;
}

}  // namespace thirdai::data