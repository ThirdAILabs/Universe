#pragma once

#include "BlockInterface.h"
#include <hashing/src/MurmurHash.h>
#include <dataset/src/encodings/count_history/CountHistoryIndex.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace thirdai::dataset {

class TrendBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * If has_count_col == false, count_col is ignored.
   */
  TrendBlock(bool has_count_col, size_t id_col, size_t timestamp_col,
             size_t count_col, uint32_t lookahead, uint32_t lookback,
             uint32_t period, std::shared_ptr<CountHistoryIndex> index,
             GraphPtr graph = nullptr, size_t max_n_neighbors = 0)
      : _lookahead_periods(lookahead / period),
        _lookback_periods(lookback / period),
        _period_seconds(period * TimeUtils::SECONDS_IN_DAY),
        _has_count_col(has_count_col),
        _id_col(id_col),
        _timestamp_col(timestamp_col),
        _count_col(count_col),
        _index(std::move(index)),
        _graph(std::move(graph)),
        _max_n_neighbors(max_n_neighbors) {
    if (_graph != nullptr && _max_n_neighbors == 0) {
      throw std::invalid_argument(
          "Provided a graph but `max_n_neighbors` is "
          "set to 0. This means "
          "graph information will not be used at all.");
    }

    _expected_num_cols = expectedNumCols();
    _index->setTimestampLifetime(_lookahead_periods + _lookback_periods);
  }

  TrendBlock(bool has_count_col, size_t id_col, size_t timestamp_col,
             size_t count_col, uint32_t lookahead, uint32_t lookback,
             uint32_t period, GraphPtr graph = nullptr,
             size_t max_n_neighbors = 0)
      : TrendBlock(has_count_col, id_col, timestamp_col, count_col, lookahead,
                   lookback, period,
                   std::make_shared<CountHistoryIndex>(
                       /* n_rows = */ 5, /* range_pow = */ 22),
                   std::move(graph), max_n_neighbors) {}

  uint32_t featureDim() const final {
    uint32_t multiplier = _max_n_neighbors + 1;
    return (_lookback_periods)*multiplier;
  };

  bool isDense() const final { return _max_n_neighbors == 0; };

  uint32_t expectedNumColumns() const final { return _expected_num_cols; };

  void prepareForBatch(const std::vector<std::string_view>& first_row) final {
    uint32_t timestamp = timestampFromInputRow(first_row);
    _index->handleLifetime(timestamp);
  }

 protected:
  std::exception_ptr buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    std::string id_str(input_row[_id_col]);
    uint32_t id = idHash(id_str);
    uint32_t timestamp = timestampFromInputRow(input_row);
    float count = countFromInputRow(input_row);
    _index->index(id, timestamp, count);

    uint32_t offset = 0;
    offset = addFeaturesForId(id, timestamp, vec, offset);
    if (_graph && _graph->count(id_str) > 0) {
      auto& neighbors = _graph->at(id_str);
      size_t included_nbrs = std::min(_max_n_neighbors, neighbors.size());
      for (size_t i = 0; i < included_nbrs; i++) {
        uint32_t neighbor_id = idHash(neighbors[i]);
        offset = addFeaturesForId(neighbor_id, timestamp, vec, offset);
      }
    }

    return nullptr;
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
    return TimeUtils::timeToEpoch(&time, 0) / _period_seconds;
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

  uint32_t addFeaturesForId(uint32_t id, uint32_t timestamp,
                            SegmentedFeatureVector& vec, uint32_t offset) {
    std::vector<float> counts(_lookback_periods);
    float mean = 0;
    fillCountsAndMean(id, timestamp, counts, mean);

    if (_lookback_periods > 1 && mean != 0) {
      center(counts, mean);
      l2Normalize(counts);
    }

    uint32_t idx = 0;
    for (const auto& count : counts) {
      if (!std::isnan(count)) {
        vec.addSparseFeatureToSegment(offset + idx, count);
      }
      idx++;
    }
    return offset + idx;
  }

  void fillCountsAndMean(uint32_t id, uint32_t timestamp,
                         std::vector<float>& counts, float& mean) {
    mean = 0;
    for (uint32_t i = 0; i < _lookback_periods; i++) {
      auto look_back = _lookahead_periods + i;
      // Prevent overflow if given a date < 1970.
      auto query_timestamp = timestamp >= look_back ? timestamp - look_back : 0;
      auto query_result = _index->query(id, query_timestamp);
      counts[i] = query_result;
      mean += std::isnan(query_result) ? 0 : query_result;
    }
    mean /= _lookback_periods;
  }

  static void center(std::vector<float>& counts, float mean) {
    for (auto& count : counts) {
      count -= mean;
    }
  }

  static void l2Normalize(std::vector<float>& counts) {
    float sum_sqr = 0;
    for (const auto& count : counts) {
      sum_sqr += std::isnan(count) ? 0 : count * count;
    }
    float l2_norm = std::sqrt(sum_sqr);
    if (l2_norm == 0) {
      return;
    }
    for (auto& count : counts) {
      count /= l2_norm;
    }
  }

  uint32_t _lookahead_periods;
  uint32_t _lookback_periods;
  uint32_t _period_seconds;
  bool _has_count_col;
  size_t _id_col;
  size_t _timestamp_col;
  size_t _count_col;
  size_t _expected_num_cols;
  std::shared_ptr<CountHistoryIndex> _index;
  GraphPtr _graph;
  size_t _max_n_neighbors;
};

}  // namespace thirdai::dataset
