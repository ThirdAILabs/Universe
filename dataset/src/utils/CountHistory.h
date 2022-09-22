#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/utils/TimeUtils.h>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace thirdai::dataset {

class CountMinSketch {
 public:
  CountMinSketch(uint32_t n_rows, uint32_t range)
      : _n_rows(n_rows),
        _range(range),
        _sketch(_n_rows * _range),
        _seeds(_n_rows) {
    for (size_t i = 0; i < _n_rows; ++i) {
      _seeds[i] = i * 314;
    }
  }

  void increment(const std::string& key, float val) {
    forEachBucket(key, [&](float& bucket) { bucket += val; });
  }

  float query(const std::string& key) {
    float min = std::numeric_limits<float>::max();
    forEachBucket(key, [&](float& bucket) { min = std::min(min, bucket); });
    return min;
  }

  void clear() { std::fill(_sketch.begin(), _sketch.end(), 0.0); }

 private:
  template <typename ForEachBucketLambdaT>
  void forEachBucket(const std::string& key, ForEachBucketLambdaT lambda) {
    for (size_t row_idx = 0; row_idx < _n_rows; ++row_idx) {
      uint32_t start_of_row = row_idx * _range;
      auto hash = hashing::MurmurHash(key.data(), key.size(), _seeds[row_idx]);
      uint32_t sketch_idx = start_of_row + (hash % _range);
      lambda(_sketch[sketch_idx]);
    }
  }

  uint32_t _n_rows;
  size_t _range;
  std::vector<float> _sketch;
  std::vector<uint32_t> _seeds;
};

class UserCountHistory {
 public:
  UserCountHistory(uint32_t lookahead_periods, uint32_t lookback_periods,
                   uint32_t period_seconds = DEFAULT_PERIOD_SECONDS,
                   uint32_t sketch_rows = 5, uint32_t sketch_range = 1 << 22)
      : _period_seconds(period_seconds),
        _lookahead_periods(lookahead_periods),
        _lookback_periods(lookback_periods),
        _start_timestamp(std::numeric_limits<int64_t>::min()),
        _recent(sketch_rows, sketch_range),
        _old(sketch_rows, sketch_range) {}

  void index(const std::string& user, int64_t timestamp, float val) {
    auto cms_key = cmsKey(user, timestampPeriods(timestamp));
    _recent.increment(cms_key, val);
  }

  std::vector<float> getHistory(const std::string& user, int64_t timestamp) {
    auto timestamp_periods = timestampPeriods(timestamp);

    std::vector<float> history(_lookback_periods);
    for (int64_t period = 0; period <= _lookback_periods; period++) {
      int64_t period_delta =
          period - _lookahead_periods - _lookback_periods + 1;
      auto cms_key = cmsKey(user, timestamp_periods + period_delta);
      history[period] = _recent.query(cms_key) + _old.query(cms_key);
    }
    return history;
  }

  void removeOutdatedCounts(int64_t timestamp) {
    if (timestamp < expiryTimestamp()) {
      return;
    }
    std::swap(_recent, _old);
    _recent.clear();
    _start_timestamp = timestamp;
  }

  static constexpr uint32_t DEFAULT_PERIOD_SECONDS = TimeObject::SECONDS_IN_DAY;

 private:
  int64_t expiryTimestamp() const {
    int64_t lifetime_periods = _lookahead_periods + _lookback_periods;
    return _start_timestamp + lifetime_periods * _period_seconds;
  }

  inline int64_t timestampPeriods(int64_t timestamp) const {
    return timestamp / _period_seconds;
  }

  static inline std::string cmsKey(const std::string& user,
                                   int64_t timestamp_periods) {
    return user + "_" + std::to_string(timestamp_periods);
  }

  uint32_t _period_seconds;
  uint32_t _lookahead_periods;
  uint32_t _lookback_periods;
  int64_t _start_timestamp;
  CountMinSketch _recent;
  CountMinSketch _old;
};

using CountHistoryPtr = std::shared_ptr<UserCountHistory>;
}  // namespace thirdai::dataset