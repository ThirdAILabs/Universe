#include <hashing/src/HashUtils.h>
#include <bitset>
#include <iostream>
#include <limits>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace thirdai::dataset {

constexpr uint32_t SECONDS_IN_DAY = 60 * 60 * 24;

struct CountMinSketch {
  CountMinSketch(size_t n_rows, uint32_t range_pow, std::vector<float>& sketch, std::vector<std::pair<uint32_t, uint32_t>>& hash_constants):
    _n_rows(n_rows), 
    _mask(std::numeric_limits<uint32_t>::max() << (32 - range_pow) >> (32 - range_pow)),
    _range(1 << range_pow), 
    _sketch_offset(sketch.size()), 
    _hash_constants_offset(hash_constants.size()), 
    _sketch(sketch),
    _hash_constants(hash_constants) {

      _sketch.resize(_sketch.size() + _n_rows * _range);
      _hash_constants.resize(_hash_constants.size() + _n_rows);
      
      for (size_t i = _hash_constants_offset; i < _hash_constants_offset + _n_rows; ++i) {
        uint32_t a = std::rand();
        if (!(a & 1)) {
          a += 1;
        }
        uint32_t b = std::rand() << range_pow >> range_pow;
        _hash_constants[i] = {a, b};
      }
    }

  void index(uint64_t x, float inc) {
    for (size_t i = 0; i < _n_rows; ++i) {
      updateIthCount(x, i, inc);
    }
  }

  float query(uint64_t x) const {
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < _n_rows; ++i) {
      std::cout << "x " << x << " ith count " << getIthCount(x, i) << std::endl;
      min = std::min(min, getIthCount(x, i));
    }
    std::cout << "min " << min << std::endl;
    return min;
  }
  
 private:
  void updateIthCount(uint64_t x, size_t i, float inc) {
    _sketch[getIthStart(i) + getIthIdx(x, i)] += inc;
  }
  
  float getIthCount(uint64_t x, size_t i) const {
    auto count = _sketch[getIthStart(i) + getIthIdx(x, i)];
    return count;
  }

  size_t getIthStart(size_t i) const {
    return _sketch_offset + i * _range;
  }

  size_t getIthIdx(uint64_t x, size_t i) const {
    const auto [a, b] = _hash_constants[_hash_constants_offset + i];
    auto before_mod = a * x + b;
    std::bitset<64> bit_before_mod(before_mod);
    
    // TODO(Geordie): Why the heck does before_mod always result in the same number...?? 
    return static_cast<uint32_t>(before_mod && _mask);
  }

  size_t _n_rows;
  size_t _mask;
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
  
  float query(uint32_t id, uint32_t start_timestamp, uint32_t range) const {
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
      std::cout << "timestamp " << start_timestamp << " day " << day << " cms_idx " << cms_idx << std::endl;
      count += _count_min_sketches[cms_idx].query(pack(id, day >> cms_idx));

      day += (1 << cms_idx);
    }
    return count;
  }  

 private:
  static uint64_t pack(uint32_t id, uint32_t timestamp) {
    return hashing::HashUtils::combineHashes(id, timestamp);
  }

  static uint32_t timestampToDay(uint32_t timestamp) { return timestamp / SECONDS_IN_DAY; }

  size_t _n_sketches = 0;
  std::vector<float> _sketch_buffer;
  std::vector<std::pair<uint32_t, uint32_t>> _hash_constants_buffer;
  std::vector<CountMinSketch> _count_min_sketches;
};
} // namespace thirdai::dataset

