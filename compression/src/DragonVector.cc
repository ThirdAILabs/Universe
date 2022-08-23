#include "DragonVector.h"
#include <hashing/src/MurmurHash.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::compression {

template <class T>
DragonVector<T>::DragonVector(const std::vector<T>& vec,
                              float compression_density, int seed_for_hashing)
    : DragonVector(vec.data(), static_cast<uint32_t>(vec.size()),
                   compression_density, seed_for_hashing) {}

template <class T>
DragonVector<T>::DragonVector(std::vector<uint32_t> indices,
                              std::vector<T> values, uint32_t original_size,
                              int seed_for_hashing)
    : _indices(std::move(indices)),
      _values(std::move(values)),
      _original_size(original_size),
      _seed_for_hashing(seed_for_hashing) {}

template <class T>
DragonVector<T>::DragonVector(const T* values, uint32_t size,
                              float compression_density, int seed_for_hashing)
    : _original_size(size),
      _compression_density(compression_density),
      _seed_for_hashing(seed_for_hashing) {
  uint32_t sketch_size =
      (std::max(static_cast<uint32_t>(compression_density * size),
                std::min(size, _min_sketch_size)));
  _indices.assign(sketch_size, 0);
  _values.assign(sketch_size, 0);

  // First calculate an approximate top-k threshold. Then, we sketch the
  // original vector to a smaller dragon vector

  T threshold = thirdai::compression::getThresholdForTopK(
      values, size, sketch_size, /*max_samples_for_random_sampling=*/100000,
      _seed_for_hashing);
  sketchVector(values, threshold, size, sketch_size);
}

template <class T>
DragonVector<T>::DragonVector(const DragonVector<T>& vec)
    : CompressedVector<T>(vec),
      _min_sketch_size(vec._min_sketch_size),
      _original_size(vec._original_size),
      _compression_density(vec._compression_density),
      _seed_for_hashing(vec._seed_for_hashing) {
  _indices.insert(std::end(_indices), std::begin(vec._indices),
                  std::end(vec._indices));
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
}

/*
 * For elements in the values array with absolute value greater than the
 * threshold, we hash the corresponding indices to a smaller _indices array and
 * store the elements in the _values array.
 */
template <class T>
void DragonVector<T>::sketchVector(const T* values, T threshold, uint32_t size,
                                   uint32_t sketch_size) {
#pragma omp parallel for default(none)                              \
    shared(_indices, _values, values, sketch_size, threshold, size, \
           _seed_for_hashing)
  for (uint32_t i = 0; i < size; i++) {
    if (std::abs(values[i]) > threshold) {
      int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                              std::to_string(i).length(),
                                              _seed_for_hashing) %
                 sketch_size;
      _indices[hash] = i;
      _values[hash] = values[i];
    }
  }
}

/*
 * Implementing std::vector's standard methods for the class
 */

/*
 * Both get and set methods should check that the dragon vector isn't empty,
 * index<_original_size. Errors thrown are similar to what std::vector would
 * thrown on incorrect accesses.
 */
template <class T>
T DragonVector<T>::get(uint32_t index) const {
  if (_indices.empty()) {
    throw std::logic_error(
        "Accessing elements from an empty compressed vector");
  }

  if (index >= _original_size) {
    throw std::out_of_range(
        "Index out of range for the compressed vector of size " +
        std::to_string(_original_size));
  }

  uint32_t sketch_size = _indices.size();
  int hash = thirdai::hashing::MurmurHash(std::to_string(index).c_str(),
                                          std::to_string(index).length(),
                                          _seed_for_hashing) %
             sketch_size;

  // If the index at the hash position is equal to index, we return the value
  // back, otherwise we return a zero. This is no-branching if-else.
  return (_indices[hash] == index) * _values[hash];
}

template <class T>
void DragonVector<T>::set(uint32_t index, T value) {
  if (_indices.empty()) {
    throw std::logic_error(
        "Incorrectly setting the index of an empty compressed vector");
  }

  if (index >= _original_size) {
    throw std::out_of_range(
        "Index out of range for the compressed vector of size " +
        std::to_string(_original_size));
  }
  uint32_t sketch_size = _indices.size();
  int hash = thirdai::hashing::MurmurHash(std::to_string(index).c_str(),
                                          std::to_string(index).length(),
                                          _seed_for_hashing) %
             sketch_size;

  _indices[hash] = index;
  _values[hash] = value;
}

template <class T>
void DragonVector<T>::assign(uint32_t size, uint32_t index, T value,
                             uint32_t original_size) {
  if (original_size != 0) {
    _original_size = original_size;
  } else {
    if (_original_size == 0) {
      std::cout
          << ("Unsafe: Do not call assign on an unintialized compressed vector "
              "without specifying the value of "
              "original vector. Original vector size is being set to size");
      _original_size = size;
    } else {
      //_original_size remains the same
      (void)1;
    }
  }

  _values.assign(size, value);
  _indices.assign(size, index);
}

template <class T>
void DragonVector<T>::clear() {
  _original_size = 0;
  _compression_density = 1;
  _values.clear();
  _indices.clear();
}

/*
 * Implementing Operator methods for the class
 */

template <class T>
DragonVector<T> DragonVector<T>::operator+(const DragonVector<T>& vec) const {
  if (_seed_for_hashing != vec._seed_for_hashing) {
    throw std::invalid_argument(
        "Seeds for hashing of the two Dragon Sketches are different. Try "
        "concatenating the sketches");
  }
  if (_indices.size() != vec._indices.size()) {
    throw std::length_error(
        "Cannot add two Dragon Sketches of different sizes");
  }

  if (_original_size != vec._original_size) {
    throw std::length_error(
        "Cannot add two Dragon Sketches with original vectors of different "
        "sizes");
  }

  std::vector<uint32_t> return_indices(_indices.size(), 0);
  std::vector<T> return_values(_indices.size(), 0);

#pragma omp parallel for default(none) \
    shared(vec, _values, _indices, return_indices, return_values)

  for (uint32_t i = 0; i < _indices.size(); i++) {
    /*
     * s=s1+s2
     * If s1[index] is non-zero, we use value and index from s1, otherwise use
     * sketch 2 Should not use if-else because of branching overheads.
     */

    return_indices[i] = _indices[i] + (_indices[i] == 0) * vec._indices[i];
    return_values[i] = _values[i] + (_indices[i] == 0) * vec._values[i];
  }
  return DragonVector(return_indices, return_values, _original_size,
                      _seed_for_hashing);
}

template <class T>
T DragonVector<T>::operator[](uint32_t index) const {
  return DragonVector<T>::get(index);
}

/*
 * Implementing utility methods for the class
 */

/*
 * Dragon vectors are not additive by default. But we can still define schemes
 * to add them up.
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

  if (_original_size != vec._original_size) {
    throw std::length_error(
        "Cannot extend a Dragon Sketch by another one with original vectors of "
        "different "
        "sizes");
  }
  _indices.insert(std::end(_indices), std::begin(vec._indices),
                  std::end(vec._indices));
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
  //_original_size remains the same
}

/*
 * Splitting a dragon vector into smaller parts. This is useful when we are
 * training in a distributed setting with ring-all-reduce framework. We need to
 * split the data into smaller parts and communicate.
 * The parameters _original_size, _seed_for_hashing remain the same for
 * the split vectors.
 */
template <class T>
std::vector<DragonVector<T>> DragonVector<T>::split(
    size_t number_chunks) const {
  if (uint32_t(number_chunks) > _indices.size()) {
    std::cout
        << "Warning: The number of chunks to split the vector is more "
           "than the size of the Dragon vector. Some chunks will be empty";
  }

  std::vector<std::vector<uint32_t>> split_indices =
      thirdai::compression::splitVector(_indices, number_chunks);
  std::vector<std::vector<T>> split_values =
      thirdai::compression::splitVector(_values, number_chunks);

  std::vector<DragonVector<T>> split_dragon;

  if (split_indices.size() != number_chunks) {
    throw std::length_error(
        "Number of vectors received after splitting is not the same as the "
        "number of chunks");
  }

  if (split_indices.size() != split_values.size()) {
    throw std::length_error(
        "Indices and Values have not been split into equal chunks");
  }

  for (size_t i = 0; i < split_indices.size(); i++) {
    if (split_indices[i].size() != split_values[i].size()) {
      throw std::length_error(
          "Size of indices and values array are not the same");
    }
    split_dragon.push_back(DragonVector(split_indices[i], split_values[i],
                                        _original_size, _seed_for_hashing));
  }
  return split_dragon;
}

/*
 * We are storing indices,values tuple hence, decompressing is just putting
 * corresponding values for the stored indices
 */
template <class T>
std::vector<T> DragonVector<T>::decompressVector() const {
  std::vector<T> decompressedVector(_original_size, 0);
  uint32_t sketch_size = static_cast<uint32_t>(_indices.size());
  for (uint32_t i = 0; i < sketch_size; i++) {
    decompressedVector[_indices[i]] += _values[i];
  }
  return decompressedVector;
}

template class DragonVector<float>;

}  // namespace thirdai::compression
