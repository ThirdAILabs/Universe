#include "CategoricalTemporal.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/transformations/Transformation.h>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

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

CategoricalTemporal::CategoricalTemporal(
    const proto::data::CategoricalTemporal& cat_temp)
    : _user_column(cat_temp.user_column()),
      _item_column(cat_temp.item_column()),
      _timestamp_column(cat_temp.timestamp_column()),
      _output_column(cat_temp.output_column()),
      _tracker_key(cat_temp.tracker_key()),
      _track_last_n(cat_temp.track_last_n()),
      _should_update_history(cat_temp.should_update_history()),
      _include_current_row(cat_temp.include_current_row()),
      _time_lag(cat_temp.time_lag()) {}

ColumnMap CategoricalTemporal::apply(ColumnMap columns, State& state) const {
  auto user_col = columns.getValueColumn<std::string>(_user_column);
  auto item_col = getItemColumn(columns);
  auto timestamp_col = columns.getValueColumn<int64_t>(_timestamp_column);

  auto& item_history_tracker = state.getItemHistoryTracker(_tracker_key);

  std::vector<std::vector<uint32_t>> last_n_items(user_col->numRows());

  for (size_t i = 0; i < user_col->numRows(); i++) {
    const std::string& user_id = user_col->value(i);
    int64_t timestamp = timestamp_col->value(i);

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

    auto& user_item_history = item_history_tracker[user_id];

    if (!user_item_history.empty() &&
        timestamp < user_item_history.back().timestamp) {
      std::stringstream error;
      error << "Expected increasing timestamps for each tracking key. Found "
               "timestamp "
            << timestamp << " after seeing timestamp "
            << user_item_history.back().timestamp << " for tracking key '"
            << user_id << "'.";
      throw std::invalid_argument(error.str());
    }

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

  auto output =
      ArrayColumn<uint32_t>::make(std::move(last_n_items), std::nullopt);
  columns.setColumn(_output_column, output);

  return columns;
}

void CategoricalTemporal::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  auto output = apply(input, state).getArrayColumn<uint32_t>(_output_column);

  for (uint32_t token : output->row(0)) {
    std::string explanation =
        "User interaction with item: " + std::to_string(token);

    explanations.store(_output_column, token, explanation);
  }
}

proto::data::Transformation* CategoricalTemporal::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* categorical_temporal = transformation->mutable_categorical_temporal();

  categorical_temporal->set_user_column(_user_column);
  categorical_temporal->set_item_column(_item_column);
  categorical_temporal->set_timestamp_column(_timestamp_column);
  categorical_temporal->set_output_column(_output_column);
  categorical_temporal->set_tracker_key(_tracker_key);

  categorical_temporal->set_track_last_n(_track_last_n);
  categorical_temporal->set_should_update_history(_should_update_history);
  categorical_temporal->set_include_current_row(_include_current_row);
  categorical_temporal->set_time_lag(_time_lag);

  return transformation;
}

}  // namespace thirdai::data
