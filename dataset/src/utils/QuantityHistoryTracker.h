#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include "CountMinSketch.h"
#include <dataset/src/utils/TimeUtils.h>
#include <utils/text/StringManipulation.h>
#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace thirdai::dataset {

enum class QuantityTrackingGranularity { Daily, Weekly, Biweekly, Monthly };

static inline QuantityTrackingGranularity stringToGranularity(
    const std::string& granularity_string) {
  auto lower_granularity_string = text::lower(granularity_string);
  if (lower_granularity_string == "daily" || lower_granularity_string == "d") {
    return dataset::QuantityTrackingGranularity::Daily;
  }
  if (lower_granularity_string == "weekly" || lower_granularity_string == "w") {
    return dataset::QuantityTrackingGranularity::Weekly;
  }
  if (lower_granularity_string == "biweekly" ||
      lower_granularity_string == "b") {
    return dataset::QuantityTrackingGranularity::Biweekly;
  }
  if (lower_granularity_string == "monthly" ||
      lower_granularity_string == "m") {
    return dataset::QuantityTrackingGranularity::Monthly;
  }
  throw std::invalid_argument(
      granularity_string +
      " is not a valid granularity option. The options are 'daily' / 'd', "
      "'weekly' / 'w', 'biweekly' / 'b', and 'monthly' / 'm',");
}

/**
 * @brief Tracks recent history of quantities associated with different
 * keys for time series modeling.
 *
 * When working with time series problems, the model needs to learn to
 * predict a quantity or class some time interval ahead using a recent
 * history of values. Thus, at any point in time, the model has access
 * to a history of values binned into time intervals that lags behind by
 * a certain amount of time.
 *
 * During training, the label that we are predicting comes from the current
 * time, so in order to have the "history" be what we expect during
 * inference, the history needs to lag behind the current timestamp.
 *
 * This data structure tracks a history of quntities of length `history_length`
 * that lags behind the current timestamp by `history_lag` intervals of time.
 * Each quantity in the history represents the sum of all records for the
 * underlying value over a time interval of size `tracking_granuality`. The
 * length of each time interval depends on the `tracking_granularity` –
 * either daily, weekly, biweekly, or monthly.
 */
class QuantityHistoryTracker {
  static constexpr uint32_t CMS_SEED = 341;

 public:
  QuantityHistoryTracker(uint32_t history_lag, uint32_t history_length,
                         QuantityTrackingGranularity tracking_granularity =
                             QuantityTrackingGranularity::Daily,
                         uint32_t sketch_rows = 5,
                         uint32_t sketch_range = 1 << 22)
      : _granularity(tracking_granularity),
        _interval_in_seconds(granularityToSeconds(tracking_granularity)),
        _history_lag(history_lag),
        _history_length(history_length),
        _start_timestamp(std::numeric_limits<int64_t>::min()),
        _recent(sketch_rows, sketch_range, CMS_SEED),
        _old(sketch_rows, sketch_range, CMS_SEED) {}

  /**
   * @brief Tells the data structure to increment the quantity tracked for
   * `key` during the time interval that contains `timestamp` by `val`.
   */
  void index(const std::string& key, int64_t timestamp, float val) {
    assert(timestamp >= _start_timestamp);
    auto cms_key = cmsKey(key, clubTimestampToInterval(timestamp));
    _recent.increment(cms_key, val);
  }

  /**
   * @brief Get a history of quantities over `history_length` intervals of time,
   * lagged behind `current_timestamp` by `history_lag` intervals of time.
   * Each value is the sum of the quantity over the time interval.
   * The length of the time interval is based on the `tracking_granularity`
   * passed to the constructor – daily, weekly, biweekly, or monthly.
   * The returned history vector is ordered from oldest to most recent.
   */
  std::vector<float> getHistory(const std::string& key,
                                int64_t current_timestamp) {
    std::vector<float> history(_history_length);

    int64_t timestamp = historyStartTimestamp(current_timestamp);

    for (uint32_t interval = 0; interval < _history_length; interval++) {
      auto cms_key = cmsKey(key, timestamp);
      history[interval] = _recent.query(cms_key) + _old.query(cms_key);
      timestamp += _interval_in_seconds;
    }
    return history;
  }

  /**
   * @brief Returns the start and end timestamps of the time interval
   * corresponding to the given position in the history returned by
   * getHistory() when given the same `current_timestamp`.
   */
  std::pair<int64_t, int64_t> getTimeRangeAtHistoryPosition(
      int64_t current_timestamp, int64_t history_pos) const {
    int64_t history_start_timestamp = historyStartTimestamp(current_timestamp);
    return {history_start_timestamp + history_pos * _interval_in_seconds,
            history_start_timestamp + (history_pos + 1) * _interval_in_seconds};
  }

  /**
   * Tells the QuantityHistoryTracker that no inputs less than the passed in
   * timestamp will be added to the tracker in the future. If the passed in
   * timestamp is more than history_lag + history_length tracking granularities
   * greater than the current lowest timestamp, the current tracked quantities
   * will be archived as old values, and the current archive will be deleted
   * permanently. This means that tracked quantities are deleted permanently
   * after two successful archivings.
   *
   * If new_lowest_timestamp is less than the current lowest timestamp, the
   * current lowest timestamp will be updated to the new lowest timestamp but
   * no archiving occurs.
   */
  void checkpoint(int64_t new_lowest_timestamp) {
    if (new_lowest_timestamp < _start_timestamp) {
      _start_timestamp = new_lowest_timestamp;
      return;
    }
    if (new_lowest_timestamp < timestampWhenSafeToRemoveOldCountSketch()) {
      return;
    }
    std::swap(_recent, _old);
    _recent.clear();
    _start_timestamp = new_lowest_timestamp;
  }

  /**
   * Clears all tracked quantities.
   */
  void reset() {
    _old.clear();
    _recent.clear();
  }

  /**
   * Returns the history lag that QuantityHistoryTracker was configured
   * with; lag is in terms of TrackingGranularities.
   */
  uint32_t historyLag() const { return _history_lag; }

  /**
   * Returns the history length that QuantityHistoryTracker was configured with;
   * length is in terms of TrackingGranularities.
   */
  uint32_t historyLength() const { return _history_length; }

  QuantityTrackingGranularity granularity() const { return _granularity; }

  static constexpr QuantityTrackingGranularity DEFAULT_TRACKING_GRANULARITY =
      QuantityTrackingGranularity::Daily;

  static auto make(uint32_t history_lag, uint32_t history_length,
                   QuantityTrackingGranularity period_seconds =
                       QuantityTrackingGranularity::Daily,
                   uint32_t sketch_rows = 5, uint32_t sketch_range = 1 << 22) {
    return std::make_shared<QuantityHistoryTracker>(
        history_lag, history_length, period_seconds, sketch_rows, sketch_range);
  }

  static uint32_t granularityToSeconds(
      QuantityTrackingGranularity granularity) {
    uint32_t n_days;
    switch (granularity) {
      case QuantityTrackingGranularity::Daily:
        n_days = 1;
        break;
      case QuantityTrackingGranularity::Weekly:
        n_days = 7;
        break;
      case QuantityTrackingGranularity::Biweekly:
        n_days = 14;
        break;
      case QuantityTrackingGranularity::Monthly:
        n_days = 30;
        break;
      default:
        throw std::invalid_argument("Invalid quanitity tracking granularity.");
    };
    return n_days * TimeObject::SECONDS_IN_DAY;
  }

 private:
  // TODO(Geordie): For clarity / to prevent accidental swapping of clubbed /
  // unclubbed timestamps, we ideally return a different type e.g.
  // ClubbedTimestamp.
  int64_t historyStartTimestamp(int64_t current_timestamp) const {
    int64_t current_timestamp_clubbed =
        clubTimestampToInterval(current_timestamp);

    // -1 at the end because a lag of 0 should include the current time interval
    int64_t start_offset_intervals = _history_lag + _history_length - 1;
    return current_timestamp_clubbed -
           start_offset_intervals * _interval_in_seconds;
  }

  int64_t timestampWhenSafeToRemoveOldCountSketch() const {
    int64_t lifetime_periods = _history_lag + _history_length;
    return _start_timestamp + lifetime_periods * _interval_in_seconds;
  }

  inline int64_t clubTimestampToInterval(int64_t timestamp) const {
    return timestamp / _interval_in_seconds * _interval_in_seconds;
  }

  static inline std::string cmsKey(const std::string& key,
                                   int64_t timestamp_periods) {
    return key + "_" + std::to_string(timestamp_periods);
  }

  QuantityTrackingGranularity _granularity;
  uint32_t _interval_in_seconds;
  uint32_t _history_lag;
  uint32_t _history_length;
  int64_t _start_timestamp;
  CountMinSketch _recent;
  CountMinSketch _old;

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  QuantityHistoryTracker() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_granularity, _interval_in_seconds, _history_lag, _history_length,
            _start_timestamp, _recent, _old);
  }
};

using QuantityHistoryTrackerPtr = std::shared_ptr<QuantityHistoryTracker>;
}  // namespace thirdai::dataset