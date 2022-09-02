#include "CountSketch.h"
#include "CompressedVector.h"
#include <hashing/src/UniversalHash.h>
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
      std::max(static_cast<uint32_t>(compression_density * size),
               static_cast<uint32_t>(1));

  // std::cout << "Inside countsketch.cc " << std::endl;
  // std::cout << "the size is: " << size
  //           << " the compression density is: " << compression_density
  //           << std::endl;
  // std::cout << "Sketch size is: " << sketch_size << std::endl;
  _count_sketches.assign(num_sketches, std::vector<T>(sketch_size, 0));
  // std::cout << "num_sketches is: " << num_sketches << std::endl;
  // std::cout << " size of hashing seed is " <<
  // _seed_for_hashing_indices.size()
  //           << " size of hashing sign is " << _seed_for_sign.size()
  //           << std::endl;
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(_seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(_seed_for_sign[i]));
  }
  // std::cout << "hashers are made" << std::endl;
  sketch(values_to_compress, size);
  // std::cout << "values are sketched now" << std::endl;
}

template <class T>
CountSketch<T>::CountSketch(std::vector<std::vector<T>> count_sketches,
                            std::vector<uint32_t> seed_for_hashing_indices,
                            std::vector<uint32_t> seed_for_sign,
                            uint32_t _uncompressed_size)
    : _count_sketches(std::move(count_sketches)),
      _seed_for_hashing_indices(std::move(seed_for_hashing_indices)),
      _seed_for_sign(std::move(seed_for_sign)),
      _uncompressed_size(_uncompressed_size) {
  // std::cout << "\n\n\n";
  // std::cout << "inside constructor using sketches\n";
  // std::cout << "size seed for hashing: " << _seed_for_hashing_indices.size()
  //           << std::endl;
  uint32_t num_sketches = static_cast<uint32_t>(_count_sketches.size());
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(_seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(_seed_for_sign[i]));
  }
  // std::cout << "hasher index size: " << _hasher_index.size()
  // << " hasher sign size: " << _hasher_sign.size() << std::endl;
  // std::cout << "\n\n\n";
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
  // std::cout << "inside get " << std::endl;
  // std::cout << "sketch size is: " << sketch_size << std::endl;
  // std::cout << "num sketches is: " << _count_sketches.size() << std::endl;
  // std::cout << "going to the for loop" << std::endl;
  // std::cout << " size of hasher index is: " << _hasher_index.size()
  //           << std::endl;
  // std::cout << "size of hasher sign is: " << _hasher_sign.size() <<
  // std::endl;
  for (size_t num_sketch = 0; num_sketch < _count_sketches.size();
       num_sketch++) {
    // std::cout << "num_sketch: " << num_sketch << std::endl;
    uint32_t hashed_index =
        _hasher_index[num_sketch].gethash(index) % sketch_size;
    uint32_t hashed_sign = _hasher_sign[num_sketch].gethash(index) % 2;

    // std::cout << " hashed_index: " << hashed_index
    // << " hashed_sign: " << hashed_sign << std::endl;
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
  return "count_sketch";
}

template <class T>
std::vector<T> CountSketch<T>::decompress() const {
  // std::cout << "inside decompression: " << std::endl;
  // std::cout << "the parameters are: uncompressed_size: " <<
  // _uncompressed_size
  //           << " num sketches: " << numSketches() << " sketchsize: " <<
  //           size()
  //           << std::endl;

  std::vector<T> decompressed_vector(_uncompressed_size, 0);
  // #pragma omp parallel for default(none)
  //     shared(decompressed_vector, _uncompressed_size)
  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    decompressed_vector[i] = get(i);
  }
  return decompressed_vector;
}

template class CountSketch<float>;
}  // namespace thirdai::compression