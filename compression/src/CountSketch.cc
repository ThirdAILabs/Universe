#include "CountSketch.h"
#include "CompressedVector.h"
#include <hashing/src/UniversalHash.h>
#include <_types/_uint32_t.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression {

template <class T>
CountSketch<T>::CountSketch(const std::vector<T>& vector_to_compress,
                            float compression_density, uint32_t num_sketches,
                            std::vector<uint32_t> seed_for_hashing_indices,
                            std::vector<uint32_t> seed_for_sign)
    : CountSketch(vector_to_compress.data(),
                  static_cast<uint32_t>(vector_to_compress.size()),
                  compression_density, num_sketches, seed_for_hashing_indices,
                  seed_for_sign) {}

template <class T>
CountSketch<T>::CountSketch(const T* values_to_compress, uint32_t size,
                            float compression_density, uint32_t num_sketches,
                            std::vector<uint32_t> seed_for_hashing_indices,
                            std::vector<uint32_t> seed_for_sign)
    : _seed_for_hashing_indices(std::move(seed_for_hashing_indices)),
      _seed_for_sign(std::move(seed_for_sign)),
      _uncompressed_size(size) {
  uint32_t sketch_size =
      std::min(static_cast<uint32_t>(compression_density * size),
               static_cast<uint32_t>(1));
  _count_sketches.assign(num_sketches, std::vector<T>(sketch_size, 0));
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(seed_for_sign[i]));
  }
  sketch(values_to_compress, size);
}

template <class T>
void CountSketch<T>::sketch(const T* values_to_compress, uint32_t size) {
  uint32_t sketch_size = static_cast<uint32_t>(_count_sketches[0].size());
  for (size_t num_sketch = 0; num_sketch < _count_sketches.size();
       num_sketch++) {
    UniversalHash hasher_index = _hasher_index[num_sketch];
    UniversalHash hasher_sign = _hasher_sign[num_sketch];
    for (uint32_t index = 0; index < size; index++) {
      uint32_t hashed_index = hasher_index.gethash(index) % sketch_size;
      uint32_t hashed_sign = hasher_sign.gethash(index) % 2;
      _count_sketches[num_sketch][hashed_index] +=
          (hashed_sign == 0) * (-values_to_compress[index]) +
          (hashed_sign == 1) * (values_to_compress[index]);
    }
  }
}

template <class T>
T CountSketch<T>::get(uint32_t index) const {
  T estimated_value = 0;
  uint32_t sketch_size = static_cast<uint32_t>(_count_sketches[0].size());
  for (size_t num_sketch = 0; num_sketch < _count_sketches.size();
       num_sketch++) {
    uint32_t hashed_index =
        _hasher_index[num_sketch].gethash(index) % sketch_size;
    uint32_t hashed_sign = _hasher_sign[num_sketch].gethash(index) % 2;

    if (hashed_sign == 0) {
      estimated_value -= _count_sketches[num_sketch][hashed_index];
    } else {
      estimated_value += _count_sketches[num_sketch][hashed_index];
    }
  }
  return estimated_value / _count_sketches.size();
}

template <class T>
void CountSketch<T>::set(uint32_t index, T value) {
  uint32_t sketch_size = static_cast<uint32_t>(_count_sketches[0].size());
  for (size_t num_sketch = 0; num_sketch < _count_sketches.size();
       num_sketch++) {
    uint32_t hashed_index =
        _hasher_index[num_sketch].gethash(index) % sketch_size;
    uint32_t hashed_sign = _hasher_sign[num_sketch].gethash(index) % 2;

    if (hashed_sign == 0) {
      _count_sketches[num_sketch][hashed_index] -= value;
    } else {
      _count_sketches[num_sketch][hashed_index] += value;
    }
  }
}

template <class T>
void CountSketch<T>::clear() {
  _count_sketches.clear();
  _seed_for_hashing_indices.clear();
  _seed_for_sign.clear();
  _hasher_sign.clear();
  _hasher_index.clear();
  _uncompressed_size = 0;
}

template <class T>
void CountSketch<T>::extend(const CountSketch<T>& other_sketch) {
  _count_sketches.insert(std::end(_count_sketches),
                         std::begin(other_sketch._count_sketches),
                         std::end(other_sketch._count_sketches));

  _seed_for_hashing_indices.insert(
      std::end(_seed_for_hashing_indices),
      std::begin(other_sketch._seed_for_hashing_indices),
      std::end(other_sketch._seed_for_hashing_indices));

  _seed_for_sign.insert(std::end(_seed_for_sign),
                        std::begin(other_sketch._seed_for_sign),
                        std::end(other_sketch._seed_for_sign));

  _hasher_index.insert(std::end(_hasher_index),
                       std::begin(other_sketch._hasher_index),
                       std::end(other_sketch._hasher_index));

  _hasher_sign.insert(std::end(_hasher_sign),
                      std::begin(other_sketch._hasher_sign),
                      std::end(other_sketch._hasher_sign));
}

// dimension of _count_sketches should be the same. We assume the user knows
// that you should never add count sketches with different underlying hash
// functions.
template <class T>
void CountSketch<T>::add(const CountSketch<T>& other_sketch) {
  uint32_t sketch_size = static_cast<uint32_t>(_count_sketches[0].size());

  for (size_t num_sketch = 0; num_sketch < _count_sketches.size();
       num_sketch++) {
    for (uint32_t i = 0; i < sketch_size; i++) {
      _count_sketches[num_sketch][i] +=
          other_sketch._count_sketches[num_sketch][i];
    }
  }
}

template <class T>
uint32_t CountSketch<T>::numSketches() const {
  return static_cast<uint32_t>(_count_sketches.size());
}

template <class T>
uint32_t CountSketch<T>::size() const {
  if (_count_sketches.empty()) {
    return 0;
  }
  return static_cast<uint32_t>(_count_sketches[0].size());
}

template <class T>
std::string CountSketch<T>::type() const {
  return "countsketch";
}

template <class T>
std::vector<T> CountSketch<T>::decompress() const {
  std::vector<T> decompressed_vector(_uncompressed_size, 0);
#pragma omp parallel for default(none) \
    shared(decompressed_vector, _uncompressed_size)
  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    decompressed_vector[i] = get(i);
  }
  return decompressed_vector;
}

template class CountSketch<float>;
}  // namespace thirdai::compression