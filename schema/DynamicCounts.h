#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <schema/InProgressVector.h>
#include <sys/types.h>
#include "Schema.h"

/**
 * TODO:
 * - srand
 * - configurable cms sizes
 * - based on counts
 * - Should I try to use circular buffer instead?
 */

namespace thirdai::schema {

struct CountMinSketch {
  CountMinSketch(size_t n_rows, uint32_t range_pow, std::vector<float>& sketch, std::vector<std::pair<uint32_t, uint32_t>>& hash_constants):
    _n_rows(n_rows), 
    _range_pow(range_pow),
    _range(1 << range_pow), 
    _sketch_offset(sketch.size()), 
    _hash_constants_offset(hash_constants.size()), 
    _sketch(sketch),
    _hash_constants(hash_constants) {

      _sketch.resize(_sketch.size() + _n_rows * _range);
      _hash_constants.resize(_hash_constants.size() + _n_rows);
      
      for (size_t i = _hash_constants_offset; i < _hash_constants_offset + _n_rows; ++i) {
        uint32_t a = rand();
        if (!(a & 1)) {
          a += 1;
        }
        uint32_t b = rand() << range_pow >> range_pow;
        _hash_constants[i] = {a, b};
      }
    }

  void index(uint64_t x, float inc) {
    for (size_t i = 0; i < _n_rows; ++i) {
      updateIthCount(x, i, inc);
    }
  }

  float query(uint64_t x) const {
    float min = MAXFLOAT;
    for (size_t i = 0; i < _n_rows; ++i) {
      min = std::min(min, getIthCount(x, i));
    }
    return min;
  }
  
 private:
  void updateIthCount(uint64_t x, size_t i, float inc) {
    _sketch[getIthStart(i) + getIthIdx(x, i)] += inc;
  }
  
  float getIthCount(uint64_t x, size_t i) const {
    return _sketch[getIthStart(i) + getIthIdx(x, i)];
  }

  size_t getIthStart(size_t i) const {
    return _sketch_offset + i * _range;
  }

  size_t getIthIdx(uint64_t x, size_t i) const {
    const auto [a, b] = _hash_constants[_hash_constants_offset + i];
    uint32_t before_mod = a * x + b;
    return before_mod >> (32 - _range_pow);
  }

  size_t _n_rows;
  size_t _range_pow;
  size_t _range;
  size_t _sketch_offset; 
  size_t _hash_constants_offset; 
  std::vector<float>& _sketch;
  std::vector<std::pair<uint32_t, uint32_t>>& _hash_constants;
};

class DynamicCounts {
  // TODO(Geordie): In this case, the CMS-are not contiguous in memory. Might want to do it differently.
  // In fact, can completely do CMS in this class. No need for separate class, right?
 public:
  explicit DynamicCounts(uint32_t max_range) {
    for (size_t largest_interval = 1; largest_interval <= max_range; largest_interval <<= 1) {
      _n_sketches++;
    }
    size_t n_buckets_pow = 24; // n_buckets ~ 1,000,000 // TODO(Geordie): this prolly needs to change. For now I just want to know the speedup.
    for (size_t i = 0; i < _n_sketches; i++) {
      _count_min_sketches.push_back(CountMinSketch(5, n_buckets_pow, _sketch_buffer, _hash_constants_buffer)); // TODO(Geordie): n rows also needs to change.
      // _interval_n_buckets >>= 1;
    }
  }

  void index(uint32_t id, uint32_t timestamp, float inc = 1.0) {
    for (size_t i = 0; i < _n_sketches; ++i) {
      auto cms_idx = i;
      auto cms_timestamp = timestampToDay(timestamp) >> i;
      _count_min_sketches[cms_idx].index(pack(id, cms_timestamp), inc);
    }
  }  
  
  uint32_t query(uint32_t id, uint32_t start_timestamp, uint32_t range) const {
    uint32_t start_day = timestampToDay(start_timestamp);
    // TODO(Geordie): Revisit
    uint32_t count = 0;
    auto day = start_day;
    auto end_day = start_day + range;
    while (day != end_day) {
      // Get highest cms_idx such that 
      // 1) day + cms interval < end_day, 
      // 2) cms_idx < n_sketches,
      // 3) interval starts on day (day has cms_idx trailing 0s).
      uint32_t next_cms_idx = 1;
      uint32_t next_interval_size = 1 << next_cms_idx;
      uint32_t next_interval_starts_on_day = !(day & (((1 << next_cms_idx) - 1)));
      while (day + next_interval_size < end_day && next_cms_idx < _n_sketches && next_interval_starts_on_day) {
        next_cms_idx++;
        next_interval_size = 1 << next_cms_idx;
        next_interval_starts_on_day = !(day & (((1 << next_cms_idx) - 1)));
      }
      uint32_t cms_idx = next_cms_idx - 1;

      count += _count_min_sketches[cms_idx].query(pack(id, day >> cms_idx));;

      day += (1 << cms_idx);
    }
    return count;
  }  

 private:
  static uint64_t pack(uint32_t id, uint32_t timestamp) {
    return static_cast<uint64_t>(id) << 32 | timestamp;
  }

  static uint32_t timestampToDay(uint32_t timestamp) { return timestamp / SECONDS_IN_DAY; }

  size_t _n_sketches = 0;
  std::vector<float> _sketch_buffer;
  std::vector<std::pair<uint32_t, uint32_t>> _hash_constants_buffer;
  std::vector<CountMinSketch> _count_min_sketches;
};

struct Window {
  Window(uint32_t lag, uint32_t size): _lag(lag), _size(size) {}
  uint32_t _lag;
  uint32_t _size;
};
struct DynamicCountsBlock: public ABlock {
  DynamicCountsBlock(std::vector<Window> window_configs, uint32_t max_window_size, uint32_t id_col, uint32_t timestamp_col, int32_t target_col, uint32_t offset, std::string timestamp_fmt)
  : _window_configs(std::move(window_configs)),
    _dc(max_window_size),
    _id_col(id_col),
    _timestamp_col(timestamp_col),
    _target_col(target_col),
    _offset(offset),
    _timestamp_fmt(std::move(timestamp_fmt)) {}

  void extractFeatures(std::vector<std::string_view> line, InProgressSparseVector& vec) override {

    uint32_t id = getNumberU32(line[_id_col]);
    
    // Get timestamp
    uint32_t timestamp = getSecondsSinceEpochU32(line[_timestamp_col], _timestamp_fmt);

    // Get target
    uint32_t target = _target_col >= 0 ? getNumberU32(line[_target_col]) : 1;
    
    // Index
    _dc.index(id, timestamp, target);
    
    // Query
    size_t i = 0;
    for (const auto& cfg : _window_configs) {
      uint32_t start_timestamp = timestamp - (cfg._lag + cfg._size - 1) * SECONDS_IN_DAY;
      float value = _dc.query(id, start_timestamp, cfg._size);
      vec.addSingleFeature(_offset + i, value);
      i++;
    }
  }

  static std::shared_ptr<ABlockConfig> Config(uint32_t id_col, uint32_t timestamp_col, int32_t target_col, std::vector<Window> window_configs, std::string timestamp_fmt) {
    return std::make_shared<DynamicCountsBlockConfig>(id_col, timestamp_col, target_col, std::move(window_configs), std::move(timestamp_fmt));
  }

  struct DynamicCountsBlockConfig: public ABlockConfig {
    DynamicCountsBlockConfig(uint32_t id_col, uint32_t timestamp_col, int32_t target_col, std::vector<Window> window_configs, std::string timestamp_fmt)
    : _window_configs(std::move(window_configs)),
      _id_col(id_col),
      _timestamp_col(timestamp_col),
      _target_col(target_col),
      _timestamp_fmt(std::move(timestamp_fmt)) {}

    std::unique_ptr<ABlock> build(uint32_t &offset) const override {
      uint32_t max_window_size = 0;
      for (const auto& cfg : _window_configs) {
        max_window_size = std::max(max_window_size, cfg._size);
      }
      auto built = std::make_unique<DynamicCountsBlock>(_window_configs, max_window_size, _id_col, _timestamp_col, _target_col, offset, _timestamp_fmt);
      offset += _window_configs.size();
      // We do not add increment based on label windows since labels are dense, 
      // and even if it wasn't, we need to use a different offset variable.
      return built;
    }



    size_t maxColumn() const override { 
      auto max_col = std::max(_id_col, _timestamp_col);
      return std::max(static_cast<int32_t>(max_col), _target_col);
    }

    size_t featureDim() const override {
      return _window_configs.size();
    }

  private:
    std::vector<Window> _window_configs;
    uint32_t _id_col;
    uint32_t _timestamp_col;
    int32_t _target_col;
    const std::string _timestamp_fmt;
  };

 private:
  std::vector<Window> _window_configs;
  DynamicCounts _dc;
  uint32_t _id_col;
  uint32_t _timestamp_col;
  int32_t _target_col;
  uint32_t _offset;
  const std::string _timestamp_fmt;
};

} // namespace thirdai::schema