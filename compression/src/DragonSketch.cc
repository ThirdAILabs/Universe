#pragma once

#include "DragonVector.h"
#include <hashing/src/MurmurHash.h>
#include <_types/_uint32_t.h>
#include <algorithm>

namespace thirdai::compression {

template <class T>
DragonVector<T>::DragonVector(const std::vector<T>& vec,
                              float compression_density, int seed_for_hashing)
    : _sketch_size(std::max(compression_density * vec.size(),
                            std::min(vec.size(), _min_sketch_size)),
                   _compression_density(compression_density),
                   _seed_for_hashing(seed_for_hashing)) {
  _indices.assign(_sketch_size, 0);
  _values.assign(_sketch_size, 0);

  float threshold = thirdai::compression::getThresholdForTopK(
      vec, _sketch_size, /*max_samples_for_random_sampling=*/100000);

  sketchVector(vec, threshold);
}

template <class T>
void DragonVector<T>::sketchVector(const std::vector<T>& vec, float threshold) {
  uint32_t loop_size = vec.size();
#pragma omp parallel for default(none)                                 \
    shared(_indices, _values, vec, _sketch_size, threshold, loop_size, \
           _seed_for_hashing)
  for (uint32_t i = 0; i < loop_size; i++) {
    if (std::abs(vec[i]) > threshold) {
      int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                              std::to_string(i).length(),
                                              _seed_for_hashing) %
                 _sketch_size;
      _indices[hash] = i;
      _values[hash] = vec[i];
    }
  }
}
}  // namespace thirdai::compression
