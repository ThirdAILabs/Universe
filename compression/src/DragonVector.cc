#include "DragonVector.h"
#include "CompressedVector.h"
#include <hashing/src/UniversalHash.h>
#include <_types/_uint32_t.h>
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
std::string DragonVector<T>::type() const {
  return "dragon";
}

/*
 * Serialization function for the dragon vector. The order of serialization is:
 * 1) Size of compression scheme string + Compression Scheme
 * 2) Uncompressed Size of the vector
 * 3) Seed for hashing
 * 4) Size of indices and values array (they are the same)
 * 5) Indices and then values array
 */

template <class T>
std::stringstream DragonVector<T>::serialize() const {
  std::stringstream output_stream;

  // Writing compression scheme (1)
  std::string compression_scheme = "dragon";
  uint32_t size = static_cast<uint32_t>(compression_scheme.size());
  output_stream.write(reinterpret_cast<char*>(&size), sizeof(uint32_t));
  output_stream.write(compression_scheme.c_str(), size);

  // Writing uncompressed size, seed_for_hashing (2,3)
  output_stream.write(reinterpret_cast<const char*>(&_uncompressed_size),
                      sizeof(uint32_t));
  output_stream.write(reinterpret_cast<const char*>(&_seed_for_hashing),
                      sizeof(uint32_t));

  // Writing size of indices, values vectors (4)
  uint32_t sketch_size = this->size();
  output_stream.write(reinterpret_cast<char*>(&sketch_size), sizeof(uint32_t));

  // Writing the indices and values vectors to the array (5)
  const uint32_t* indices_data = _indices.data();
  output_stream.write(reinterpret_cast<const char*>(indices_data),
                      sizeof(uint32_t) * _indices.size());

  const T* values_data = _values.data();
  output_stream.write(reinterpret_cast<const char*>(values_data),
                      sizeof(T) * _values.size());
  return output_stream;
}

template <class T>
DragonVector<T>::DragonVector(std::stringstream& input_stream) {
  // Reading the compression scheme (1)
  uint32_t string_size;
  std::string compression_scheme;
  input_stream.read(reinterpret_cast<char*>(&string_size), sizeof(uint32_t));
  char* buff(new char[string_size]);
  input_stream.read(reinterpret_cast<char*>(buff), string_size);
  compression_scheme.assign(buff, string_size);

  // Reading uncompressed size, seed_for_hashing (2,3)
  uint32_t uncompressed_size;
  uint32_t seed_for_hashing;
  input_stream.read(reinterpret_cast<char*>(&uncompressed_size),
                    sizeof(uint32_t));
  input_stream.read(reinterpret_cast<char*>(&seed_for_hashing),
                    sizeof(uint32_t));

  _uncompressed_size = uncompressed_size;  // NOLINT
  _seed_for_hashing = seed_for_hashing;    // NOLINT

  // Reading size of indices and values array (4)
  uint32_t sketch_size;
  input_stream.read(reinterpret_cast<char*>(&sketch_size), sizeof(uint32_t));
  _indices.resize(sketch_size);
  _values.resize(sketch_size);

  input_stream.read(reinterpret_cast<char*>(_indices.data()),
                    sizeof(uint32_t) * sketch_size);
  input_stream.read(reinterpret_cast<char*>(_values.data()),
                    sizeof(T) * sketch_size);
}

template class DragonVector<float>;

}  // namespace thirdai::compression
