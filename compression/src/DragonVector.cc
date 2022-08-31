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
                              std::vector<T> values, uint32_t original_size,
                              uint32_t seed_for_hashing)
    : _indices(std::move(indices)),
      _values(std::move(values)),
      _original_size(original_size),
      _seed_for_hashing(seed_for_hashing) {}

template <class T>
DragonVector<T>::DragonVector(const T* values_to_compress, uint32_t size,
                              float compression_density,
                              uint32_t seed_for_hashing,
                              uint32_t sample_population_size)
    : _original_size(size),
      _compression_density(compression_density),
      _seed_for_hashing(seed_for_hashing) {
  uint32_t sketch_size =
      std::max(static_cast<uint32_t>(compression_density * size),
               std::min(size, _min_sketch_size));

  _indices.assign(sketch_size, 0);
  _values.assign(sketch_size, 0);

  // First calculate an approximate top-k threshold. Then, we sketch the
  // original vector to a smaller dragon vector

  T threshold = thresholdForTopK(values_to_compress, size, compression_density,
                                 /*seed_for_sampling=*/seed_for_hashing,
                                 sample_population_size);
  sketch(values_to_compress, threshold, size, sketch_size);
}

/*
 * For elements in the values array with absolute value greater than the
 * threshold, we hash the corresponding indices to a smaller _indices array and
 * store the elements in the _values array.
 */
template <class T>
void DragonVector<T>::sketch(const T* values, T threshold, uint32_t size,
                             uint32_t sketch_size) {
  UniversalHash hash_function = UniversalHash(_seed_for_hashing);
#pragma omp parallel for default(none)                              \
    shared(_indices, _values, values, sketch_size, threshold, size, \
           _seed_for_hashing, hash_function)

  for (uint32_t i = 0; i < size; i++) {
    if (std::abs(values[i]) > threshold) {
      uint32_t hash = hash_function.gethash(i) % sketch_size;
      _indices[hash] = i;
      _values[hash] = values[i];
    }
  }
}

/*
 * Implementing std::vector's standard methods for the class
 */

template <class T>
T DragonVector<T>::get(uint32_t index) const {
  uint32_t sketch_size = _indices.size();
  UniversalHash hash_function = UniversalHash(_seed_for_hashing);
  uint32_t hash = hash_function.gethash(index) % sketch_size;
  // If the index at the hash position is equal to index, we return the value
  // back, otherwise we return a zero. This is no-branching if-else.
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
  _original_size = 0;
  _compression_density = 1;
  _values.clear();
  _indices.clear();
}

/*
 * Implementing utility methods for the class
 */

template <class T>
bool DragonVector<T>::isAdditive() const {
  return false;
}

template <class T>
void DragonVector<T>::extend(const DragonVector<T>& vec) {
  /*
   * We should not check whether the seeds for hashing are the same for the two
   * Dragon vectors. We will directly append the indices and values of given
   * vector to the current one but leave all other parameters intact
   */
  _indices.insert(std::end(_indices), std::begin(vec._indices),
                  std::end(vec._indices));
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
  //_original_size remains the same
}

template <class T>
std::vector<T> DragonVector<T>::decompress() const {
  std::vector<T> decompressedVector(_original_size, 0);
  uint32_t sketch_size = static_cast<uint32_t>(_indices.size());
  for (uint32_t i = 0; i < sketch_size; i++) {
    decompressedVector[_indices[i]] += _values[i];
  }
  return decompressedVector;
}

template <class T>
std::string DragonVector<T>::type() const {
  return "dragon";
}

template class DragonVector<float>;

}  // namespace thirdai::compression
