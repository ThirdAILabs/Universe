#include "CountSketch.h"
#include "CompressedVector.h"
#include "Serializer.h"
#include <hashing/src/UniversalHash.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression {

template <class T>
CountSketch<T>::CountSketch(
    const std::vector<T>& vector_to_compress, float compression_density,
    uint32_t num_sketches,
    const std::vector<uint32_t>& seed_for_hashing_indices,
    const std::vector<uint32_t>& seed_for_sign)
    : CountSketch(vector_to_compress.data(),
                  static_cast<uint32_t>(vector_to_compress.size()),
                  compression_density, num_sketches, (seed_for_hashing_indices),
                  (seed_for_sign)) {}

template <class T>
CountSketch<T>::CountSketch(
    const T* values_to_compress, uint32_t size, float compression_density,
    uint32_t num_sketches,
    const std::vector<uint32_t>& seed_for_hashing_indices,
    const std::vector<uint32_t>& seed_for_sign)
    : _uncompressed_size(size) {
  uint32_t sketch_size =
      std::max(static_cast<uint32_t>(compression_density * size),
               static_cast<uint32_t>(1));
  /*
   * TODO(Shubh):  Forcing sketch size to be odd was giving better accuracies
   * than even sketch sizes.  Hence, sketch sizes are forced to be odd. We need
   * to look more into why this happens.
   */
  if (num_sketches <= 0) {
    throw std::invalid_argument(
        "Atleast one sketch is needed for Count Sketching the values");
  }
  sketch_size = sketch_size % 2 == 0 ? sketch_size + 1 : sketch_size;
  _count_sketches.assign(num_sketches, std::vector<T>(sketch_size, 0));
  for (uint32_t i = 0; i < num_sketches; i++) {
    _hasher_index.push_back(UniversalHash(seed_for_hashing_indices[i]));
    _hasher_sign.push_back(UniversalHash(seed_for_sign[i]));
  }
  sketch(values_to_compress, size);
}

template <class T>
void CountSketch<T>::sketch(const T* values_to_compress, uint32_t size) {
#pragma omp parallel for default(none) shared(size, values_to_compress)
  for (uint32_t index = 0; index < size; index++) {
    set(index, values_to_compress[index]);
  }
}

template <class T>
int CountSketch<T>::hash_sign(uint32_t sketch_id, uint32_t index) const {
  // If _hasher_sign returns 0, we return -1, else 1
  int sign = 2 * (_hasher_sign[sketch_id].gethash(index) % 2) - 1;
  return sign;
}

template <class T>
uint32_t CountSketch<T>::hash_index(uint32_t sketch_id, uint32_t index) const {
  uint32_t sketch_size = this->size();
  return _hasher_index[sketch_id].gethash(index) % sketch_size;
}

template <class T>
T CountSketch<T>::get(uint32_t index) const {
  T estimated_value = 0;
  for (uint32_t sketch_id = 0; sketch_id < numSketches(); sketch_id++) {
    uint32_t hashed_index = hash_index(sketch_id, index);
    int hashed_sign = hash_sign(sketch_id, index);
    estimated_value += hashed_sign * _count_sketches[sketch_id][hashed_index];
  }
  return estimated_value / _count_sketches.size();
}

template <class T>
void CountSketch<T>::set(uint32_t index, T value) {
  for (uint32_t sketch_id = 0; sketch_id < numSketches(); sketch_id++) {
    uint32_t hashed_index = hash_index(sketch_id, index);
    int hashed_sign = hash_sign(sketch_id, index);
    _count_sketches[sketch_id][hashed_index] += hashed_sign * value;
  }
}

template <class T>
void CountSketch<T>::clear() {
  _count_sketches.clear();
  _hasher_sign.clear();
  _hasher_index.clear();
  _uncompressed_size = 0;
}

template <class T>
void CountSketch<T>::extend(const CountSketch<T>& other_sketch) {
  if (other_sketch.size() != this->size()) {
    throw std::length_error(
        "Cannot extend a count sketch with another count sketch of different "
        "size.");
  }
  _count_sketches.insert(std::end(_count_sketches),
                         std::begin(other_sketch._count_sketches),
                         std::end(other_sketch._count_sketches));

  _hasher_index.insert(std::end(_hasher_index),
                       std::begin(other_sketch._hasher_index),
                       std::end(other_sketch._hasher_index));

  _hasher_sign.insert(std::end(_hasher_sign),
                      std::begin(other_sketch._hasher_sign),
                      std::end(other_sketch._hasher_sign));
}

// Dimension of _count_sketches should be the same. Adding count sketches with
// different underlying hash functions will give weird results.
template <class T>
void CountSketch<T>::add(const CountSketch<T>& other_sketch) {
  uint32_t sketch_size = static_cast<uint32_t>(_count_sketches[0].size());

  if (other_sketch.numSketches() != this->numSketches()) {
    throw std::length_error(
        "Cannot add count sketches that have different number of sketch "
        "vectors.");
  }

  if (other_sketch.size() != this->size()) {
    throw std::length_error("Cannot add count sketches of different sizes.");
  }
  // TODO(Shubh): Parallelize this.
  for (uint32_t sketch_id = 0; sketch_id < numSketches(); sketch_id++) {
    for (uint32_t i = 0; i < sketch_size; i++) {
      _count_sketches[sketch_id][i] +=
          other_sketch._count_sketches[sketch_id][i];
    }
  }
}

template <class T>
void CountSketch<T>::divide(uint32_t divisor) {
  if (divisor == 0) {
    throw std::invalid_argument("Cannot divide a Count Sketch by 0");
  }
  uint32_t num_sketches = numSketches();
#pragma omp parallel for default(none) shared(num_sketches, divisor)
  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    for (uint32_t i = 0; i < _count_sketches[0].size(); i++) {
      _count_sketches[sketch_id][i] /= divisor;
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

  // Writing seed for hashing indices (4)
  std::vector<uint32_t> seed_for_hashing_indices;
  seed_for_hashing_indices.reserve(num_sketches);
  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    seed_for_hashing_indices.emplace_back(_hasher_index[sketch_id].seed());
  }
  outputHelper.writeVector(seed_for_hashing_indices);

  // Writing seed for sign (5)
  std::vector<uint32_t> seed_for_sign;
  seed_for_sign.reserve(num_sketches);
  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    seed_for_sign.emplace_back(_hasher_sign[sketch_id].seed());
  }
  outputHelper.writeVector(seed_for_sign);

  // Writing Count Sketch Vectors (6)
  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    outputHelper.writeVector(_count_sketches[sketch_id]);
  }
}

template <class T>
CountSketch<T>::CountSketch(const char* serialized_data) {
  serializer::BinaryInputHelper inputHelper(serialized_data);

  // Reading the compression scheme (1)
  uint32_t compression_scheme;
  inputHelper.read(&compression_scheme);

  // Reading uncompressed_size (2)
  inputHelper.read(&_uncompressed_size);

  // Reading number of count sketches (3)
  uint32_t num_sketches;
  inputHelper.read(&num_sketches);

  // Reading seed for hashing indices (4)
  std::vector<uint32_t> seed_for_hashing_indices;
  seed_for_hashing_indices.reserve(num_sketches);
  inputHelper.readVector(seed_for_hashing_indices);

  // Reading seed for sign (5)
  std::vector<uint32_t> seed_for_sign;
  seed_for_sign.reserve(num_sketches);
  inputHelper.readVector(seed_for_sign);

  // Reading Count Sketch Vectors (6)
  _count_sketches.resize(num_sketches);
  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    inputHelper.readVector(_count_sketches[sketch_id]);
  }

  for (uint32_t sketch_id = 0; sketch_id < num_sketches; sketch_id++) {
    _hasher_index.push_back(UniversalHash(seed_for_hashing_indices[sketch_id]));
    _hasher_sign.push_back(UniversalHash(seed_for_sign[sketch_id]));
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