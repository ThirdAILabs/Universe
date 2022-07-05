#pragma once

#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace thirdai::dataset {

class CountMinSketch {
 public:
  CountMinSketch(uint32_t n_rows, uint32_t range_pow)
      : _n_rows(n_rows),
        _mask(computeMask(range_pow)),
        _range(1 << range_pow),
        _sketch(_n_rows * _range),
        _seeds(_n_rows) {
    for (size_t i = 0; i < _n_rows; ++i) {
      _seeds[i] = i * 314;
    }
  }

  void index(uint64_t x, float inc) {
    for (size_t i = 0; i < _n_rows; ++i) {
      _sketch[startOfRow(i) + indexInRow(i, x)] += inc;
    }
  }

  float query(uint64_t x) const {
    float min = std::numeric_limits<float>::max();
    for (size_t i = 0; i < _n_rows; ++i) {
      auto count = _sketch[startOfRow(i) + indexInRow(i, x)];
      min = std::min(min, count);
    }
    return min;
  }

  void clear() { std::fill(_sketch.begin(), _sketch.end(), 0.0); }

 private:
  uint32_t startOfRow(uint32_t i) const { return i * _range; }

  uint32_t indexInRow(uint32_t i, uint64_t x) const {
    void* x_ptr = static_cast<void*>(&x);
    auto hash =
        hashing::MurmurHash(static_cast<char*>(x_ptr), sizeof(x), _seeds[i]);
    return hash & _mask;
  }

  static uint32_t computeMask(size_t range_pow) {
    uint32_t ones = std::numeric_limits<uint32_t>::max();
    uint32_t shift_bits = 32 - range_pow;
    return ones << shift_bits >> shift_bits;
  }

  uint32_t _n_rows;
  uint32_t _mask;
  size_t _range;
  std::vector<float> _sketch;
  std::vector<uint32_t> _seeds;
};
}  // namespace thirdai::dataset