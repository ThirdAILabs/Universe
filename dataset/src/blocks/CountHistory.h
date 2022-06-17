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
  Window (uint32_t lag, uint32_t size)
    : lag(lag), size(size) {}
 
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
  CountHistoryBlock(bool has_count_col, uint32_t id_col, uint32_t timestamp_col,
                    uint32_t count_col, std::vector<Window> windows,
                    DynamicCountsConfig& index_config,bool has_near = false, std::vector<std::vector< std::string>> near_neighbours = {{" "}})
      : _primary_start_timestamp(0),
        _has_count_col(has_count_col),
        _has_near(has_near),
        _near_neighbours(std::move(near_neighbours)),
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

    _matching = get_allocate(_near_neighbours);
  }

  static std::unordered_map<std::string_view,uint32_t> get_allocate(std::vector<std::vector< std::string>> neighbours) {
    std::unordered_map<std::string_view,uint32_t> maps;
    for(uint32_t i=0;i<neighbours.size();i++) {
      //std::string_view temp = std::string_view(neighbours[i][0].c_str(),neighbours[i][0].size());
      maps[neighbours[i][0]] = i;
      //maps[temp] = i;
    }
    return maps;
  }

  uint32_t featureDim() const final { return _has_near? _windows.size()*7 : _windows.size(); };

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
    std::vector<uint32_t> ids(6);
    if(_has_near) {
      int t = _matching.find(id_str)->second;
      for(int i=0;i<6;i++) {
        auto id_str = _near_neighbours[t][i];
        std::from_chars(id_str.data(), id_str.data() + id_str.size(), ids[i]);
      }
    }

#pragma omp critical
    {
      if (timestamp - _primary_start_timestamp > _lifetime) {
        _primary_start_timestamp = timestamp;
        _index.handleNewLifetime();
      }
    }
    _index.index(id, timestamp, count);
    uint32_t k = 0;
    for (uint32_t i = 0; i < _windows.size(); i++) {
      const auto& [lag, size] = _windows[i];
      // Prevent overflow if given a date < 1970.
      auto look_back = (lag + size - 1) * SECONDS_IN_DAY;
      auto query_timestamp = timestamp >= look_back ? timestamp - look_back : 0;
      auto query_result =
          _index.query(id, query_timestamp, size);
      vec.addSparseFeatureToSegment(k++, query_result);
      if(_has_near) {
        for(uint32_t j=0 ;j<6;j++) {
          auto query_result =
          _index.query(ids[j], query_timestamp, size);
      vec.addSparseFeatureToSegment(k++, query_result);
        }
      }
    }
  }

 private:
  uint32_t _lifetime;
  uint32_t _primary_start_timestamp;
  bool _has_count_col;
  bool _has_near;
  std::vector<std::vector< std::string>> _near_neighbours;
  uint32_t _id_col;
  uint32_t _timestamp_col;
  uint32_t _count_col;
  uint32_t _expected_num_cols;
  std::vector<Window> _windows;
  DynamicCounts _index;
  std::unordered_map<std::string_view,uint32_t> _matching;
};

}  // namespace thirdai::dataset
