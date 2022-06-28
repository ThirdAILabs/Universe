#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <dataset/src/utils/TimeUtils.h>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace thirdai::dataset {

class TrendBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * If has_count_col == false, count_col is ignored.
   */
  TrendBlock(bool has_count_col, size_t id_col, size_t timestamp_col,
             size_t count_col, size_t horizon, size_t lookback,
             DynamicCountsConfig& index_config)
      : _primary_start_timestamp(0),
        _horizon(horizon),
        _lookback(lookback),
        _has_count_col(has_count_col),
        _id_col(id_col),
        _timestamp_col(timestamp_col),
        _count_col(count_col),
        _index(index_config) {
    uint32_t max_history_days = lookback + horizon;
    _lifetime = max_history_days * SECONDS_IN_DAY;

    size_t max_col_idx = 0;
    max_col_idx = std::max(max_col_idx, _id_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    if (_has_count_col) {
      max_col_idx = std::max(max_col_idx, _count_col);
    }
    _expected_num_cols = max_col_idx + 1;

    assert(_lookback != 0);
  }

  uint32_t featureDim() const final { return _lookback + 1; };

  bool isDense() const final { return true; };

  uint32_t expectedNumColumns() const final { return _expected_num_cols; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    auto id_str = input_row[_id_col];
    uint32_t id{};
    std::from_chars(id_str.data(), id_str.data() + id_str.size(), id);

    std::tm time = TimeUtils::timeStringToTimeObject(input_row[_timestamp_col]);
    // TODO(Geordie) should timestamp be uint64_t?
    uint32_t timestamp = std::mktime(&time);

    float count = 1.0;
    if (_has_count_col) {
      auto count_str = input_row[_count_col];
      char* end;
      count = std::strtof(count_str.data(), &end);
    }

#pragma omp critical
    {
      if (timestamp - _primary_start_timestamp > _lifetime) {
        _primary_start_timestamp = timestamp;
        _index.handleNewLifetime();
      }
    }
    _index.index(id, timestamp, count);
    std::vector<float> counts(_lookback);
    float sum = 0;
    for (uint32_t i = 0; i < _lookback; i++) {
      const auto lag = _horizon + i;
      auto look_back = lag * SECONDS_IN_DAY;
      // Prevent overflow if given a date < 1970.
      auto query_timestamp = timestamp >= look_back ? timestamp - look_back : 0;
      auto query_result = _index.query(id, query_timestamp, 1);
      assert(query_result >= 0);
      counts[i] = query_result;
      sum += query_result;
    }

    float mean = sum / _lookback;
    if (sum != 0) {
      for (auto& count : counts) {
        count = (count - mean) / mean;
      }
    }
    for (const auto& count : counts) {
      vec.addDenseFeatureToSegment(count);
    }
    vec.addDenseFeatureToSegment(mean);
  }

 private:
  uint32_t _lifetime;
  uint32_t _primary_start_timestamp;
  size_t _horizon;
  size_t _lookback;
  bool _has_count_col;
  size_t _id_col;
  size_t _timestamp_col;
  size_t _count_col;
  size_t _expected_num_cols;
  DynamicCounts _index;
};

}  // namespace thirdai::dataset
