#include "DragonVector.h"
#include "CompressedVector.h"
#include <hashing/src/UniversalHash.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cstring>

using UniversalHash = thirdai::hashing::UniversalHash;

namespace thirdai::compression {

template <class T>
DragonVector<T>::DragonVector(const std::vector<T>& vector_to_compress,
                              float compression_density,
                              uint32_t seed_for_hashing,
                              uint32_t sample_population_size)
    : DragonVector(vector_to_compress.data(),
                   static_cast<uint32_t>(vector_to_compress.size()),
                   compression_density, seed_for_hashing,
                   sample_population_size) {}

template <class T>
DragonVector<T>::DragonVector(std::vector<uint32_t> indices,
                              std::vector<T> values, uint32_t uncompressed_size,
                              uint32_t seed_for_hashing)
    : _indices(std::move(indices)),
      _values(std::move(values)),
      _uncompressed_size(uncompressed_size),
      _seed_for_hashing(seed_for_hashing) {}

template <class T>
DragonVector<T>::DragonVector(const T* values_to_compress, uint32_t size,
                              float compression_density,
                              uint32_t seed_for_hashing,
                              uint32_t sample_population_size)
    : _uncompressed_size(size), _seed_for_hashing(seed_for_hashing) {
  uint32_t sketch_size =
      std::max(static_cast<uint32_t>(compression_density * size),
               std::min(size, _min_sketch_size));

  _indices.assign(sketch_size, 0);
  _values.assign(sketch_size, 0);

  /*
   * The routine first calculates an approximate top-k threshold. Then, it
   * sketches the original vector to a smaller dragon vector. It sketches only
   * the values which are larger than the threshold.
   */
  T estimated_threshold = estimateTopKThreshold(
      values_to_compress, size, compression_density,
      /*seed_for_sampling=*/seed_for_hashing, sample_population_size);
  sketch(values_to_compress, estimated_threshold, size, sketch_size);
}

/*
 * The methods hashes the (index, value) pair in the original array to the
 * vectors (_indices, _values) if the value is larger than the threshold.
 */
template <class T>
void DragonVector<T>::sketch(const T* values, T threshold, uint32_t size,
                             uint32_t sketch_size) {
  UniversalHash hash_function = UniversalHash(_seed_for_hashing);

  /*
   * TODO(TSK-567): MSVC complains about sharing values in the below block.
   * Disabling short term to get builds green.
   *    D:\a\Universe\Universe\compression\src\DragonVector.cc(68,9): error
   *    C3028: 'thirdai::compression::DragonVector<float>::_values': only a
   *    variable or static data member can be used in a data-sharing clause
   *    [D:\a\Universe\Universe\build\_thirdai.vcxproj]
   */
#pragma omp parallel for default(none) \
    shared(values, sketch_size, threshold, size, hash_function)

  for (uint32_t i = 0; i < size; i++) {
    if (std::abs(values[i]) > threshold) {
      uint32_t hash = hash_function.gethash(i) % sketch_size;
      _indices[hash] = i;
      _values[hash] = values[i];
    }
  }
}

/*
 * If the index at the hash position is equal to index, we return the value
 * back, otherwise we return a zero because the index is not stored in the
 * Dragon Vector
 */
template <class T>
T DragonVector<T>::get(uint32_t index) const {
  uint32_t sketch_size = _indices.size();
  UniversalHash hash_function = UniversalHash(_seed_for_hashing);
  uint32_t hash = hash_function.gethash(index) % sketch_size;
  return (_indices[hash] == index) * _values[hash];
}

template <class T>
void DragonVector<T>::set(uint32_t index, T value) {
  uint32_t sketch_size = _indices.size();
  UniversalHash hash_function = UniversalHash(_seed_for_hashing);
  uint32_t hash = hash_function.gethash(index) % sketch_size;
  _indices[hash] = index;
  _values[hash] = value;
}

template <class T>
void DragonVector<T>::clear() {
  _uncompressed_size = 0;
  _values.clear();
  _indices.clear();
}

/*
 * Implementing utility methods for the class
 */

template <class T>
void DragonVector<T>::extend(const DragonVector<T>& vec) {
  /*
   * NOTE: Do not call get function on a Dragon Vector which has been extended
   * by another one. On extending a Dragon Sketch, the sketch size changes which
   * means that we cannot use get function on this modified sketch.
   * We do not need to check whether the seeds for hashing are the same for the
   * two Dragon vectors since we will directly append the indices and values of
   * given vector to the current one and leave all other parameters intact.
   * Extend is non-lossy, we do not lose any information about indices,values
   * even when we add Dragon Vectors with different seeds.
   */
  _indices.insert(std::end(_indices), std::begin(vec._indices),
                  std::end(vec._indices));
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
  //_uncompressed_size remains the same
}

/*
 * We are storing indices,values tuple hence, decompressing is just
 * putting corresponding values for the stored indices
 */
template <class T>
std::vector<T> DragonVector<T>::decompress() const {
  std::vector<T> decompressedVector(_uncompressed_size, 0);
  uint32_t sketch_size = static_cast<uint32_t>(_indices.size());
  for (uint32_t i = 0; i < sketch_size; i++) {
    decompressedVector[_indices[i]] += _values[i];
  }
  return decompressedVector;
}

template <class T>
CompressionScheme DragonVector<T>::type() const {
  return CompressionScheme::Dragon;
}

/*
 * The order of serialization for dragon vector is as follows:
 * 1) An enum representing compression scheme
 * 2) Uncompressed Size of the vector
 * 3) Seed for hashing
 * 4) Size of indices and values array (they are the same)
 * 5) Indices and then values array
 */
template <class T>
void DragonVector<T>::serialize(char* serialized_data) const {
  size_t curr_pos = 0;

  // Writing compression scheme (1)
  uint32_t compression_scheme = static_cast<uint32_t>(type());
  std::memcpy(serialized_data, reinterpret_cast<char*>(&compression_scheme),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing uncompressed size, seed_for_hashing (2,3)
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(&_uncompressed_size),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(&_seed_for_hashing),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing size of indices, values vectors (4)
  uint32_t sketch_size = this->size();
  std::memcpy(serialized_data + curr_pos, reinterpret_cast<char*>(&sketch_size),
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Writing the indices and values vectors to the array (5)
  const uint32_t* indices_data = _indices.data();
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(indices_data),
              sizeof(uint32_t) * _indices.size());
  curr_pos += sizeof(uint32_t) * _indices.size();
  const T* values_data = _values.data();
  std::memcpy(serialized_data + curr_pos,
              reinterpret_cast<const char*>(values_data),
              sizeof(T) * _values.size());
  curr_pos += sizeof(T) * _values.size();
}

template <class T>
DragonVector<T>::DragonVector(const char* serialized_data) {
  size_t curr_pos = 0;

  // Reading the compression scheme (1)
  uint32_t compression_scheme;

  std::memcpy(reinterpret_cast<char*>(&compression_scheme),
              serialized_data + curr_pos, sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  // Reading uncompressed size, seed_for_hashing (2,3)
  uint32_t uncompressed_size;
  uint32_t seed_for_hashing;
  std::memcpy(reinterpret_cast<char*>(&uncompressed_size),
              serialized_data + curr_pos, sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  std::memcpy(reinterpret_cast<char*>(&seed_for_hashing),
              serialized_data + curr_pos, sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);

  _uncompressed_size = uncompressed_size;  // NOLINT
  _seed_for_hashing = seed_for_hashing;    // NOLINT

  // Reading size of indices and values array (4)
  uint32_t sketch_size;
  std::memcpy(reinterpret_cast<char*>(&sketch_size), serialized_data + curr_pos,
              sizeof(uint32_t));
  curr_pos += sizeof(uint32_t);
  _indices.resize(sketch_size);
  _values.resize(sketch_size);

  // Reading the indices and the values array (5)
  std::memcpy(reinterpret_cast<char*>(_indices.data()),
              serialized_data + curr_pos, sizeof(uint32_t) * sketch_size);
  curr_pos += sizeof(uint32_t) * sketch_size;
  std::memcpy(reinterpret_cast<char*>(_values.data()),
              serialized_data + curr_pos, sizeof(T) * sketch_size);
  curr_pos += sizeof(uint32_t) * sketch_size;
}

template <class T>
uint32_t DragonVector<T>::serialized_size() const {
  uint32_t serialized_size = 0;
  // Compression scheme (1)
  std::string compression_scheme = "dragon";
  serialized_size +=
      sizeof(uint32_t) + sizeof(char) * compression_scheme.size();

  // Uncompressed size, seed_for_hashing (2,3)
  serialized_size += 2 * sizeof(uint32_t);

  // Size of indices and values array (4)
  serialized_size += sizeof(uint32_t);

  // The indices and the values array (5)
  serialized_size += _indices.size() * (sizeof(uint32_t) + sizeof(T));
  return serialized_size;
}

template class DragonVector<float>;

}  // namespace thirdai::compression
