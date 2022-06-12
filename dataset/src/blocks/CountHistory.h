#pragma once

#include "BlockInterface.h"
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <dataset/src/utils/TimeUtils.h>
#include <charconv>
#include <cstdlib>
#include <ctime>

namespace thirdai::dataset {

struct Window {
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
  CountHistoryBlock(bool has_count_col, uint32_t id_col, uint32_t timestamp_col, uint32_t count_col, std::vector<Window> windows, DynamicCountsConfig& index_config)
      : _has_count_col(has_count_col), _id_col(id_col), _timestamp_col(timestamp_col), _count_col(count_col), _windows(std::move(windows)), _index(index_config) {
    
    uint32_t max_col_idx = 0;
    max_col_idx = std::max(max_col_idx, _id_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    if (_has_count_col) {
      max_col_idx = std::max(max_col_idx, _count_col);
    }
    _expected_num_cols = max_col_idx + 1;
  }

  uint32_t featureDim() const final { return _windows.size(); };

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
    
    _index.index(id, timestamp, count);
    for (uint32_t i = 0; i < _windows.size(); i++) {
      const auto& [lag, size] = _windows[i];
      auto query_result = _index.query(id, timestamp - (lag + size) * SECONDS_IN_DAY, size);
      vec.addSparseFeatureToSegment(i, query_result);
    }
  }

 private:

  bool _has_count_col;
  uint32_t _id_col;
  uint32_t _timestamp_col;
  uint32_t _count_col;
  uint32_t _expected_num_cols;
  std::vector<Window> _windows;
  DynamicCounts _index;
};

} // namespace thirdai::dataset
