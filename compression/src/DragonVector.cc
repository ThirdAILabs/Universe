#include "DragonVector.h"
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
DragonVector<T>::DragonVector(const std::vector<T>& vector_to_compress,
                              float compression_density,
                              uint32_t seed_for_hashing,
                              uint32_t sample_population_size)
    : DragonVector(vector_to_compress.data(),
                   static_cast<uint32_t>(vector_to_compress.size()),
                   compression_density, seed_for_hashing,
                   sample_population_size) {}

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
  UniversalHash hash_function(_seed_for_hashing);

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
  UniversalHash hash_function(_seed_for_hashing);
  uint32_t hash = hash_function.gethash(index) % sketch_size;
  return (_indices[hash] == index) * _values[hash];
}

template <class T>
void DragonVector<T>::set(uint32_t index, T value) {
  uint32_t sketch_size = _indices.size();
  UniversalHash hash_function(_seed_for_hashing);
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
 * DragonVectors with different underlying hash functions should not be added.
 * Similarly, adding two DragonVectors with different underlying uncompressed
 * vectors has random behavior. Not throwing exceptions but caller should be
 * wary of these facts.
 *
 * NOTE: Adding two DragonVectors is lossy in nature. In
 * current implementation, we keep the index that has the highest value. add is
 * lossy in nature but does not increase the memory footprint.
 *
 * NOTE: DragonVector class is modelled exactly after the DRAGONN research
 * paper. Approximately 40-45% of a DragonVector obtained after compressing a
 * vector is empty i.e., only 50-60% of the DragonVector has non-zero indices
 * and values. Thus, adding 2-3 such DragonVectors would mean that the number of
 * non-zeros go up and hence, when training in a compressed setting, it is
 * advisable to add DragonVectors when the number of nodes is not high. For
 * example, in a tree all reduce setting, add DragonVectors for the levaes and
 * then concat.
 */
template <class T>
void DragonVector<T>::add(const DragonVector<T>& vec) {
  if (vec.size() != this->size()) {
    throw std::length_error("Cannot add two DragonVectors of different sizes");
  }
  uint32_t sketch_size = this->size();
#pragma omp parallel for default(none) shared(sketch_size, vec)
  for (uint32_t index = 0; index < sketch_size; index++) {
    if (std::abs(vec._values[index]) > std::abs(_values[index])) {
      _indices[index] = vec._indices[index];
      _values[index] = vec._values[index];
    }
  }
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
 * 4) Indices and then values array
 * While writing vectors, we first write the size and then the contents.
 */
template <class T>
void DragonVector<T>::serialize(char* serialized_data) const {
  serializer::BinaryOutputHelper outputHelper(serialized_data);

  // Writing compression scheme (1)
  uint32_t compression_scheme = static_cast<uint32_t>(type());
  outputHelper.write(&compression_scheme);

  // Writing uncompressed size, seed_for_hashing (2,3)
  outputHelper.write(&_uncompressed_size);
  outputHelper.write(&_seed_for_hashing);

  // Writing indices and values vectors (4)
  outputHelper.writeVector(_indices);
  outputHelper.writeVector(_values);
}

template <class T>
DragonVector<T>::DragonVector(const char* serialized_data) {
  serializer::BinaryInputHelper inputHelper(serialized_data);

  // Reading the compression scheme (1)
  uint32_t compression_scheme;
  inputHelper.read(&compression_scheme);

  // Reading uncompressed size, seed_for_hashing (2,3)
  uint32_t uncompressed_size;
  uint32_t seed_for_hashing;
  inputHelper.read(&uncompressed_size);
  inputHelper.read(&seed_for_hashing);

  _uncompressed_size = uncompressed_size;  // NOLINT
  _seed_for_hashing = seed_for_hashing;    // NOLINT

  // Reading indices and values array (4)
  inputHelper.readVector(_indices);
  inputHelper.readVector(_values);
}

template <class T>
uint32_t DragonVector<T>::serialized_size() const {
  uint32_t serialized_size = 0;
  // Compression scheme (1)
  serialized_size += sizeof(uint32_t);

  // Uncompressed size, seed_for_hashing (2,3)
  serialized_size += 2 * sizeof(uint32_t);

  // Size of indices array (4). We first write the size and then the elements
  serialized_size += sizeof(uint32_t) + _indices.size() * sizeof(uint32_t);

  // Size of values array (5)
  serialized_size += sizeof(uint32_t) + _values.size() * sizeof(T);
  return serialized_size;
}

template class DragonVector<float>;

}  // namespace thirdai::compression
