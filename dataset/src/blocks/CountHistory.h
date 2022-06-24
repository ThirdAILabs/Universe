#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <dataset/src/utils/TimeUtils.h>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <ctime>
#include <string>
#include <utility>

namespace thirdai::dataset {

struct Window {
  Window(uint32_t lag, uint32_t size) : lag(lag), size(size) {}

  uint32_t lag;
  uint32_t size;
};

class CountHistoryBlock : public Block {
 public:
  /**
   * Constructor.
   *
   * If has_count_col == false, count_col is ignored.
   */
  CountHistoryBlock(
      bool has_count_col, uint32_t id_col, uint32_t timestamp_col,
      uint32_t count_col, std::vector<Window> windows,
      DynamicCountsConfig& index_config, bool has_near_neighbours = false,
      const std::vector<std::vector<std::string>>& adjacency_matrix = {{}})
      /*The expected format of adjacency matrix is first column is id of the node,
      rest are ids of most correlated nodes.*/
      : _primary_start_timestamp(0),
        _has_count_col(has_count_col),
        _has_near_neighbours(has_near_neighbours),
        _id_col(id_col),
        _timestamp_col(timestamp_col),
        _count_col(count_col),
        _windows(std::move(windows)),
        _index(index_config) {
    uint32_t max_history_days = 0;
    for (const auto& window : _windows) {
      max_history_days = std::max(max_history_days, window.lag + window.size);
    }
    _lifetime = max_history_days * SECONDS_IN_DAY;

    uint32_t max_col_idx = 0;
    max_col_idx = std::max(max_col_idx, _id_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    if (_has_count_col) {
      max_col_idx = std::max(max_col_idx, _count_col);
    }
    _expected_num_cols = max_col_idx + 1;

    if (_has_near_neighbours) {
      _num_neighbours = adjacency_matrix[0].size();
      _map_id_to_near_neighbours = mapNeighbours(adjacency_matrix);
    }
  }

  static std::unordered_map<uint32_t, std::vector<uint32_t>> mapNeighbours(
      const std::vector<std::vector<std::string>>& near_neighbours) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> maps;
    for (const auto& neighbours : near_neighbours) {
      std::vector<uint32_t> res;
      for (auto neighbour : neighbours) {
        uint32_t id{};
        std::from_chars(neighbour.data(), neighbour.data() + neighbour.size(),
                        id);
        res.push_back(id);
      }
      maps[res[0]] = std::vector<uint32_t>(res.begin() + 1, res.end());
    }
    return maps;
  }

  uint32_t featureDim() const final {
    return _has_near_neighbours ? _windows.size() * (_num_neighbours)
                                : _windows.size();
  };

  bool isDense() const final { return false; };

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
    std::vector<uint32_t> ids;
    if (_has_near_neighbours) {
      ids = _map_id_to_near_neighbours[id];
    }

#pragma omp critical
    {
      if (timestamp - _primary_start_timestamp > _lifetime) {
        _primary_start_timestamp = timestamp;
        _index.handleNewLifetime();
      }
    }
    _index.index(id, timestamp, count);
    uint32_t position = 0;
    for (auto window : _windows) {
      const auto& [lag, size] = window;
      // Prevent overflow if given a date < 1970.
      auto look_back = (lag + size - 1) * SECONDS_IN_DAY;
      auto query_timestamp = timestamp >= look_back ? timestamp - look_back : 0;
      auto query_result = _index.query(id, query_timestamp, size);
      vec.addSparseFeatureToSegment(position++, query_result);
      if (_has_near_neighbours) {
        for (auto id : ids) {
          auto query_result = _index.query(id, query_timestamp, size);
          vec.addSparseFeatureToSegment(position++, query_result);
        }
      }
    }
  }

 private:
  uint32_t _lifetime;
  uint32_t _primary_start_timestamp;
  bool _has_count_col;
  bool _has_near_neighbours;
  uint32_t _num_neighbours;
  uint32_t _id_col;
  uint32_t _timestamp_col;
  uint32_t _count_col;
  uint32_t _expected_num_cols;
  std::vector<Window> _windows;
  DynamicCounts _index;
  std::unordered_map<uint32_t, std::vector<uint32_t>> _map_id_to_near_neighbours;
};

}  // namespace thirdai::dataset
