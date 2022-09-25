#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/MurmurHash.h>
#include <limits>
#include <string>
#include <vector>

namespace thirdai::dataset {

class QuantityHistoryTracker;

class CountMinSketch {
  friend QuantityHistoryTracker;

 public:
  CountMinSketch(uint32_t n_rows, uint32_t range, uint32_t seed)
      : _n_rows(n_rows),
        _range(range),
        _sketch(_n_rows * _range),
        _seeds(_n_rows) {
    for (size_t i = 0; i < _n_rows; ++i) {
      _seeds[i] = i * seed;
    }
  }

  void increment(const std::string& key, float val) {
    forEachBucket(key, [&](float& bucket) { bucket += val; });
  }

  float query(const std::string& key) {
    float min = std::numeric_limits<float>::max();
    forEachBucket(key, [&](float& bucket) { min = std::min(min, bucket); });
    return min;
  }

  void clear() { std::fill(_sketch.begin(), _sketch.end(), 0.0); }

 private:
  template <typename ForEachBucketLambdaT>
  void forEachBucket(const std::string& key, ForEachBucketLambdaT lambda) {
    static_assert(std::is_convertible<ForEachBucketLambdaT,
                                      std::function<void(float&)>>::value);

    for (size_t row_idx = 0; row_idx < _n_rows; ++row_idx) {
      uint32_t start_of_row = row_idx * _range;
      auto hash = hashing::MurmurHash(key.data(), key.size(), _seeds[row_idx]);
      uint32_t sketch_idx = start_of_row + (hash % _range);
      lambda(_sketch[sketch_idx]);
    }
  }

  uint32_t _n_rows;
  size_t _range;
  std::vector<float> _sketch;
  std::vector<uint32_t> _seeds;

  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  CountMinSketch() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_n_rows, _range, _sketch, _seeds);
  }
};

}  // namespace thirdai::dataset