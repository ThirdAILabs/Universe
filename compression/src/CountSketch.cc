#include "CountSketch.h"
#include "CompressedVector.h"
#include "Serializer.h"
#include <hashing/src/UniversalHash.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
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

  sketch_size = sketch_size % 2 == 0 ? sketch_size + 1 : sketch_size;
  _count_sketches.assign(num_sketches, std::vector<T>(sketch_size, 0));
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(_seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(_seed_for_sign[i]));
  }
  sketch(values_to_compress, size);
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
  uint32_t num_sketches = static_cast<uint32_t>(_count_sketches.size());
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(_seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(_seed_for_sign[i]));
  }
}

template <class T>
void CountSketch<T>::sketch(const T* values_to_compress, uint32_t size) {
  for (uint32_t index = 0; index < size; index++) {
    set(index, values_to_compress[index]);
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
CompressionScheme CountSketch<T>::type() const {
  return CompressionScheme::CountSketch;
}

template <class T>
std::vector<T> CountSketch<T>::decompress() const {
  std::vector<T> decompressed_vector(_uncompressed_size, 0);
#pragma omp parallel for default(none) shared(decompressed_vector)
  for (uint32_t i = 0; i < _uncompressed_size; i++) {
    decompressed_vector[i] = get(i);
  }
  return decompressed_vector;
}

/*
 * The order of serialization for count sketch is as follows:
 * 1) An enum for compression scheme
 * 2) Uncompressed Size of the vector
 * 3) Number of count_sketches
 * 4) Seeds for hashing indices
 * 5) Seeds for sign
 * 6) Count Sketch Vectors
 * While writing vectors, we first write the size and then the contents.
 */
template <class T>
void CountSketch<T>::serialize(char* serialized_data) const {
  serializer::BinaryOutputHelper outputHelper(serialized_data);

  // Writing compression scheme (1)
  uint32_t compression_scheme = static_cast<uint32_t>(type());
  outputHelper.write(&compression_scheme);

  // Writing uncompressed size (2)
  outputHelper.write(&_uncompressed_size);

  // Writing number of count sketches (3)
  uint32_t num_sketches = numSketches();
  outputHelper.write(&num_sketches);

  // Writing Seeds for hashing indices (4)
  outputHelper.writeVector(_seed_for_hashing_indices);

  // Writing Seeds for sign (5)
  outputHelper.writeVector(_seed_for_sign);

  // Writing Count Sketch Vectors (7)
  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    outputHelper.writeVector(_count_sketches[num_sketch]);
  }
}

template <class T>
CountSketch<T>::CountSketch(const char* serialized_data) {
  serializer::BinaryInputHelper inputHelper(serialized_data);

  // Reading the compression scheme (1)
  uint32_t compression_scheme;
  inputHelper.read(&compression_scheme);

  // Reading uncompressed_size (2)
  uint32_t uncompressed_size;
  inputHelper.read(&uncompressed_size);
  _uncompressed_size = uncompressed_size;  // NOLINT

  // Reading number of count sketches (3)
  uint32_t num_sketches;
  inputHelper.read(&num_sketches);

  // Reading seed for hashing indices (4)
  inputHelper.readVector(_seed_for_hashing_indices);

  // Reading seed for sign (5)
  inputHelper.readVector(_seed_for_sign);

  // Reading Count Sketch Vectors (7)
  _count_sketches.resize(num_sketches);
  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    inputHelper.readVector(_count_sketches[num_sketch]);
  }

  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    _hasher_index.push_back(
        UniversalHash(_seed_for_hashing_indices[num_sketch]));
    _hasher_sign.push_back(UniversalHash(_seed_for_sign[num_sketch]));
  }
}

template <class T>
uint32_t CountSketch<T>::serialized_size() const {
  uint32_t serialized_size = 0;
  // Compression scheme (1)
  serialized_size += sizeof(uint32_t);

  // Uncompressed size (2)
  serialized_size += sizeof(uint32_t);

  // Number of count sketches
  serialized_size += sizeof(uint32_t);

  // Seeds for hashing indices
  serialized_size += sizeof(uint32_t) + sizeof(uint32_t) * numSketches();

  // Seeds for hashing indices
  serialized_size += sizeof(uint32_t) + sizeof(uint32_t) * numSketches();

  // CountSketch Vectors
  serialized_size += numSketches() * (sizeof(uint32_t) + sizeof(T) * size());
  return serialized_size;
}

template class CountSketch<float>;
}  // namespace thirdai::compression