#pragma once

#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace thirdai::dataset {

struct SketchMemory {
  std::vector<float> sketch;
  std::vector<uint32_t> hash_seeds;
};

class CountMinSketch {
 public:
  CountMinSketch(uint32_t n_rows, uint32_t range_pow,
                 std::vector<float>& sketch, std::vector<uint32_t>& hash_seeds)
      : _n_rows(n_rows),
        _mask(computeMask(range_pow)),
        _range(1 << range_pow),
        _sketch_offset(sketch.size()),
        _hash_seeds_offset(hash_seeds.size()),
        _sketch(sketch),
        _hash_seeds(hash_seeds) {
    _sketch.resize(_sketch.size() + _n_rows * _range);
    _hash_seeds.resize(_hash_seeds.size() + _n_rows);

    for (size_t i = _hash_seeds_offset; i < _hash_seeds_offset + _n_rows; ++i) {
      _hash_seeds[i] = i * 314;
    }
  }

  CountMinSketch(uint32_t n_rows, uint32_t range_pow,
                 SketchMemory& sketch_memory)
      : CountMinSketch(n_rows, range_pow, sketch_memory.sketch,
                       sketch_memory.hash_seeds) {}

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

  void clear() {
    auto sketch_start = _sketch.begin() + _sketch_offset;
    std::fill(sketch_start, sketch_start + (_range * _n_rows), 0.0);
  }

 private:
  uint32_t startOfRow(uint32_t i) const { return _sketch_offset + i * _range; }

  uint32_t indexInRow(uint32_t i, uint64_t x) const {
    void* x_ptr = static_cast<void*>(&x);
    auto hash = hashing::MurmurHash(static_cast<char*>(x_ptr), sizeof(x),
                                    _hash_seeds[i]);
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
  uint32_t _sketch_offset;
  uint32_t _hash_seeds_offset;
  std::vector<float>& _sketch;
  std::vector<uint32_t>& _hash_seeds;
};
}  // namespace thirdai::dataset