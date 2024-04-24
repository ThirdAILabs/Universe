#include "NumericalTemporal.h"
#include <data/src/columns/ArrayColumns.h>
#include <stdexcept>

namespace thirdai::data {

NumericalTemporal::NumericalTemporal(
    std::string user_column, std::string value_column,
    std::string timestamp_column, std::string output_column,
    std::string tracker_key, size_t history_len, int64_t interval_len,
    bool should_update_history, bool include_current_row, int64_t time_lag)
    : _user_column(std::move(user_column)),
      _value_column(std::move(value_column)),
      _timestamp_column(std::move(timestamp_column)),
      _output_column(std::move(output_column)),
      _tracker_key(std::move(tracker_key)),
      _history_len(history_len),
      _interval_len(interval_len),
      _should_update_history(should_update_history),
      _include_current_row(include_current_row),
      _time_lag(time_lag) {
  if (time_lag < 0) {
    throw std::invalid_argument("time_lag must be > 0.");
  }
  if (_include_current_row && time_lag != 0) {
    throw std::invalid_argument(
        "time_lag must be 0 if include_current_row=True.");
  }
}

void center(std::vector<float>& counts) {
  float mean = 0.0;
  for (auto count : counts) {
    mean += count;
  }
  mean /= counts.size();
  for (auto& count : counts) {
    count -= mean;
  }
}

void l2normalize(std::vector<float>& counts) {
  float sum_of_squares = 0.0;
  for (auto count : counts) {
    sum_of_squares += count * count;
  }
  if (sum_of_squares == 0.0) {
    return;
  }
  float l2norm = std::sqrt(sum_of_squares);
  for (auto& count : counts) {
    count /= l2norm;
  }
}

ColumnMap NumericalTemporal::apply(ColumnMap columns, State& state) const {
  auto user_col = columns.getValueColumn<std::string>(_user_column);
  auto value_col = getValueColumn(columns);
  auto timestamp_col = columns.getValueColumn<int64_t>(_timestamp_column);

  auto& count_history_tracker = state.getCountHistoryTracker(_tracker_key);

  std::vector<std::vector<float>> last_n_counts(user_col->numRows());

  for (size_t i = 0; i < columns.numRows(); i++) {
    const std::string& user_id = user_col->value(i);
    int64_t curr_interval = interval(timestamp_col->value(i));

    std::vector<float> user_last_n_counts(_history_len, 0);

    if (_include_current_row && _time_lag == 0) {
      user_last_n_counts.back() += value_col->value(i);
    }

    auto& user_history = count_history_tracker[user_id];

    if (!user_history.empty() && curr_interval < user_history.back().interval) {
      int64_t last_interval = user_history.back().interval;
      std::stringstream error;
      error << "Expected increasing timestamps for each tracking key. Found "
               "timestamp in the interval ["
            << curr_interval * _interval_len << ", "
            << (curr_interval + 1) * _interval_len
            << ") after seeing timestamp in the interval ["
            << last_interval * _interval_len << ", "
            << (last_interval + 1) * _interval_len << ") for tracking key '"
            << user_id << "'.";
      throw std::invalid_argument(error.str());
    }

    // The -1 is so that the current interval is included. For example if the
    // history length is 2 and the time lag is 0, then we want the previous and
    // current intervals.
    int64_t start_interval = curr_interval - (_history_len + _time_lag - 1);
    int64_t end_interval = start_interval + _history_len;

    size_t relevant_intervals = 0;
    for (auto it = user_history.rbegin(); it != user_history.rend(); ++it) {
      if (it->interval < start_interval) {
        break;
      }
      if (it->interval < end_interval) {
        user_last_n_counts.at(it->interval - start_interval) += it->value;
      }
      relevant_intervals++;
    }

    if (_should_update_history) {
      size_t n_outdated = user_history.size() - relevant_intervals;
      user_history.erase(user_history.begin(),
                         user_history.begin() + n_outdated);

      if (!user_history.empty() &&
          user_history.back().interval == curr_interval) {
        user_history.back().value += value_col->value(i);
      } else {
        user_history.push_back({value_col->value(i), curr_interval});
      }
    }

    center(user_last_n_counts);
    l2normalize(user_last_n_counts);

    last_n_counts[i] = std::move(user_last_n_counts);
  }

  auto output =
      ArrayColumn<float>::make(std::move(last_n_counts), std::nullopt);
  columns.setColumn(_output_column, output);

  return columns;
}

}  // namespace thirdai::data