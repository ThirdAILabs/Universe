#pragma once

#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/encodings/count_history/DailyCountHistoryIndex.h>
#include <dataset/src/utils/TimeUtils.h>
#include <cstdlib>

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
             GraphPtr graph = nullptr, size_t max_n_neighbors = 0)
      : _lifetime((lookback + horizon) * SECONDS_IN_DAY),
        _horizon(horizon),
        _lookback(lookback),
        _has_count_col(has_count_col),
        _id_col(id_col),
        _timestamp_col(timestamp_col),
        _count_col(count_col),
        _index(/* n_rows = */ 5, /* range_pow = */ 22,
               /* lifetime = */ _lifetime),
        _graph(std::move(graph)),
        _max_n_neighbors(max_n_neighbors) {
    if (_graph != nullptr && _max_n_neighbors == 0) {
      throw std::invalid_argument(
          "[SequentialClassifier] Provided a graph but `max_n_neighbors` is "
          "set to 0. This means "
          "graph information will not be used at all.");
    }

    _expected_num_cols = expectedNumCols();
  }

  uint32_t featureDim() const final {
    uint32_t multiplier = _max_n_neighbors + 1;
    return (_lookback + 1) * multiplier;
  };

  bool isDense() const final { return _max_n_neighbors == 0; };

  uint32_t expectedNumColumns() const final { return _expected_num_cols; };

  void prepareForBatch(const std::vector<std::string_view>& first_row) final {
    std::tm time = TimeUtils::timeStringToTimeObject(first_row[_timestamp_col]);
    // TODO(Geordie) should timestamp be uint64_t?
    uint32_t timestamp = TimeUtils::timeToEpoch(&time, 0);
    _index.handleLifetime(timestamp);
  }

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    std::string id_str(input_row[_id_col]);
    uint32_t id = idHash(id_str);
    uint32_t timestamp = timestampFromInputRow(input_row);
    float count = countFromInputRow(input_row);
    _index.index(id, timestamp, count);

    addFeaturesForId(id, timestamp, vec);
    if (_graph && _graph->count(id_str) > 0) {
      auto& neighbors = _graph->at(id_str);
      size_t included_nbrs = std::min(_max_n_neighbors, neighbors.size());
      for (size_t i = 0; i < included_nbrs; i++) {
        uint32_t neighbor_id = idHash(neighbors[i]);
        addFeaturesForId(neighbor_id, timestamp, vec);
      }
    }
  }

 private:
  uint32_t expectedNumCols() const {
    size_t max_col_idx = 0;
    max_col_idx = std::max(max_col_idx, _id_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    if (_has_count_col) {
      max_col_idx = std::max(max_col_idx, _count_col);
    }

    return max_col_idx + 1;
  }

  static uint32_t idHash(const std::string_view id_str) {
    const char* start = id_str.data();
    uint32_t len = id_str.size();
    return hashing::MurmurHash(start, len, /* seed = */ 341);
  }

  uint32_t timestampFromInputRow(
      const std::vector<std::string_view>& input_row) const {
    std::tm time = TimeUtils::timeStringToTimeObject(input_row[_timestamp_col]);
    return TimeUtils::timeToEpoch(&time, 0);
  }

  float countFromInputRow(
      const std::vector<std::string_view>& input_row) const {
    float count = 1.0;
    if (_has_count_col) {
      auto count_str = input_row[_count_col];
      char* end;
      count = std::strtof(count_str.data(), &end);
    }
    return count;
  }

  void addFeaturesForId(uint32_t id, uint32_t timestamp,
                        SegmentedFeatureVector& vec) {
    std::vector<float> counts(_lookback);
    float sum = 0;
    fillCountsAndSum(id, timestamp, counts, sum);

    /*
      Center and normalize by sum so sum is 0 and
      values are always between -1 and 1.
    */
    float mean = sum / _lookback;
    centerAndNormalize(counts, sum, mean);

    for (const auto& count : counts) {
      vec.addDenseFeatureToSegment(count);
    }
    vec.addDenseFeatureToSegment(mean);
  }

  void fillCountsAndSum(uint32_t id, uint32_t timestamp,
                        std::vector<float>& counts, float& sum) {
    for (uint32_t i = 0; i < _lookback; i++) {
      auto look_back = (_horizon + i) * SECONDS_IN_DAY;
      // Prevent overflow if given a date < 1970.
      auto query_timestamp = timestamp >= look_back ? timestamp - look_back : 0;
      auto query_result = _index.query(id, query_timestamp);
      assert(query_result >= 0);
      counts[i] = query_result;
      sum += query_result;
    }
  }

  static void centerAndNormalize(std::vector<float>& counts, float sum,
                                 float mean) {
    if (sum != 0) {
      for (auto& count : counts) {
        count = (count - mean) / sum;
      }
    }
  }

  uint32_t _lifetime;
  size_t _horizon;
  size_t _lookback;
  bool _has_count_col;
  size_t _id_col;
  size_t _timestamp_col;
  size_t _count_col;
  size_t _expected_num_cols;
  DailyCountHistoryIndex _index;
  GraphPtr _graph;
  size_t _max_n_neighbors;
};

}  // namespace thirdai::dataset
