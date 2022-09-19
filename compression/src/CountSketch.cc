#include "CountSketch.h"
#include "CompressedVector.h"
#include <hashing/src/UniversalHash.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>
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
std::string CountSketch<T>::type() const {
  return "count_sketch";
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
 * Serialization function for the dragon vector. The order of serialization is:
 * 1) Size of compression scheme string + Compression Scheme
 * 2) Uncompressed Size of the vector
 * 3) Number of count_sketches
 * 4) Seeds for hashing indices
 * 5) Seeds for sign
 * 6) Size of each of the count sketch (same for each count sketch)
 * 7) Count Sketch Vectors
 */
template <class T>
void CountSketch<T>::serialize(char* serialized_data) const {
  size_t curr_pos = 0;

  // Writing compression scheme (1)
  std::string compression_scheme = "count_sketch";
  uint32_t size = static_cast<uint32_t>(compression_scheme.size());
  std::memcpy(serialized_data, reinterpret_cast<char*>(&size),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);
  std::memcpy(serialized_data + curr_pos, compression_scheme.c_str(), size);
  curr_pos += size;

  // Writing uncompressed size (2)
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(&_uncompressed_size),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing number of count sketches (3)
  uint32_t num_sketches = numSketches();
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<char*>(&num_sketches), sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing Seeds for hashing indices (4)
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(_seed_for_hashing_indices.data()),
              sizeof(uint32_t) * num_sketches);
  curr_pos += sizeof(uint32_t) * num_sketches;

  // Writing Seeds for sign (5)
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(_seed_for_sign.data()),
              sizeof(uint32_t) * num_sketches);
  curr_pos += sizeof(uint32_t) * num_sketches;

  // Writing size of count sketch (6)
  uint32_t sketch_size = this->size();
  std::memcpy(serialized_data + curr_pos, reinterpret_cast<char*>(&sketch_size),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing Count Sketch Vectors (7)
  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    std::memcpy(
        serialized_data + curr_pos,
        reinterpret_cast<const char*>(_count_sketches[num_sketch].data()),
        sizeof(T) * sketch_size);
    curr_pos += sizeof(T) * sketch_size;
  }
}

template <class T>
CountSketch<T>::CountSketch(const char* serialized_data) {
  size_t curr_pos = 0;

  // Reading the compression scheme (1)
  uint32_t string_size;
  std::string compression_scheme;
  std::memcpy(reinterpret_cast<char*>(&string_size), serialized_data + curr_pos,
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);
  char* buff(new char[string_size]);
  std::memcpy(reinterpret_cast<char*>(buff), serialized_data + curr_pos,
              string_size);
  curr_pos += string_size;
  compression_scheme.assign(buff, string_size);

  // Reading uncompressed_size (2)
  uint32_t uncompressed_size;
  std::memcpy(reinterpret_cast<char*>(&uncompressed_size),
              serialized_data + curr_pos, sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);
  _uncompressed_size = uncompressed_size;  // NOLINT

  // Reading number of count sketches (3)
  uint32_t num_sketches;
  std::memcpy(reinterpret_cast<char*>(&num_sketches),
              serialized_data + curr_pos, sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Reading seed for hashing indices (4)
  _seed_for_hashing_indices.resize(num_sketches);
  std::memcpy(reinterpret_cast<char*>(_seed_for_hashing_indices.data()),
              serialized_data + curr_pos, sizeof(uint32_t) * num_sketches);
  curr_pos += sizeof(uint32_t) * num_sketches;

  // Reading seed for sign (5)
  _seed_for_sign.resize(num_sketches);
  std::memcpy(reinterpret_cast<char*>(_seed_for_sign.data()),
              serialized_data + curr_pos, sizeof(uint32_t) * num_sketches);
  curr_pos += sizeof(uint32_t) * num_sketches;

  // Reading size of count_sketch (6)
  uint32_t sketch_size = this->size();
  std::memcpy(reinterpret_cast<char*>(&sketch_size), serialized_data + curr_pos,
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Reading Count Sketch Vectors (7)
  _count_sketches.resize(num_sketches);
  for (uint32_t num_sketch = 0; num_sketch < num_sketches; num_sketch++) {
    _count_sketches[num_sketch].resize(sketch_size);
    std::memcpy(reinterpret_cast<char*>(_count_sketches[num_sketch].data()),
                serialized_data + curr_pos, sizeof(T) * sketch_size);
    curr_pos += sizeof(T) * sketch_size;
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
  std::string compression_scheme = "count_sketch";
  serialized_size +=
      sizeof(uint32_t) + sizeof(char) * compression_scheme.size();

  // Uncompressed_size (2)
  serialized_size += sizeof(uint32_t);

  // Number of count sketches
  serialized_size += sizeof(uint32_t);

  // Seeds for hashing indices
  serialized_size += sizeof(uint32_t) * numSketches();

  // Seeds for hashing sign
  serialized_size += sizeof(uint32_t) * numSketches();

  // Size of count sketch
  serialized_size += sizeof(uint32_t);

  // Count Sketch Vectors
  serialized_size += sizeof(T) * size() * numSketches();

  return serialized_size;
}

template class CountSketch<float>;
}  // namespace thirdai::compression