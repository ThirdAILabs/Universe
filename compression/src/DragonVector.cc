#include "DragonVector.h"
#include <hashing/src/MurmurHash.h>
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

namespace thirdai::compression {

template <class T>
DragonVector<T>::DragonVector(const std::vector<T>& vec,
                              float compression_density, int seed_for_hashing)
    : _sketch_size(std::max(uint32_t(compression_density * vec.size()),
                            std::min(uint32_t(vec.size()), _min_sketch_size))),
      _original_size(uint32_t(vec.size())),
      _compression_density(compression_density),
      _seed_for_hashing(seed_for_hashing) {
  _indices.assign(_sketch_size, 0);
  _values.assign(_sketch_size, 0);

  T threshold = thirdai::compression::getThresholdForTopK(
      vec, _sketch_size, /*max_samples_for_random_sampling=*/100000);

  sketchVector(vec, threshold);
}

template <class T>
DragonVector<T>::DragonVector(std::vector<uint32_t> indices,
                              std::vector<T> values, uint32_t size,
                              uint32_t original_size, int seed_for_hashing)
    : _indices(std::move(indices)),
      _values(std::move(values)),
      _sketch_size(size),
      _original_size(original_size),
      _seed_for_hashing(seed_for_hashing) {}

template <class T>
DragonVector<T>::DragonVector(const T* values, float compression_density,
                              uint32_t size, int seed_for_hashing)
    : _sketch_size(std::max(uint32_t(compression_density * size),
                            std::min(size, _min_sketch_size))),
      _original_size(size),
      _compression_density(compression_density),
      _seed_for_hashing(seed_for_hashing) {
  _indices.assign(_sketch_size, 0);
  _values.assign(_sketch_size, 0);

  T threshold = thirdai::compression::getThresholdForTopK(
      values, size, _sketch_size, /*max_samples_for_random_sampling=*/100000);
  sketchVector(values, threshold, size);
}

template <class T>
void DragonVector<T>::sketchVector(const std::vector<T>& vec, T threshold) {
  uint32_t loop_size = vec.size();
#pragma omp parallel for default(none)                                 \
    shared(_indices, _values, vec, _sketch_size, threshold, loop_size, \
           _seed_for_hashing)
  for (uint32_t i = 0; i < loop_size; i++) {
    if (std::abs(vec[i]) > threshold) {
      int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                              std::to_string(i).length(),
                                              _seed_for_hashing) %
                 _sketch_size;
      _indices[hash] = i;
      _values[hash] = vec[i];
    }
  }
}

template <class T>
void DragonVector<T>::sketchVector(const T* values, T threshold,
                                   uint32_t size) {
#pragma omp parallel for default(none)                               \
    shared(_indices, _values, values, _sketch_size, threshold, size, \
           _seed_for_hashing)
  for (uint32_t i = 0; i < size; i++) {
    if (std::abs(values[i]) > threshold) {
      int hash = thirdai::hashing::MurmurHash(std::to_string(i).c_str(),
                                              std::to_string(i).length(),
                                              _seed_for_hashing) %
                 _sketch_size;
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
  if (_sketch_size == 0) {
    throw std::logic_error(
        "Accessing elements from an empty compressed vector");
  }

  if (index >= _original_size) {
    throw std::out_of_range(
        "Index out of range for the compressed vector of size " +
        std::to_string(_original_size));
  }

  int hash = thirdai::hashing::MurmurHash(std::to_string(index).c_str(),
                                          std::to_string(index).length(),
                                          _seed_for_hashing) %
             _sketch_size;

  // if the index at the hash position is equal to index, we return hash back,
  // otherwise we return a zero. no branching if-else
  return (_indices[hash] == index) * _values[hash];
}

template <class T>
void DragonVector<T>::set(uint32_t index, T value) {
  if (_sketch_size == 0) {
    throw std::logic_error(
        "Incorrectly setting the index of an empty compressed vector");
  }

  if (index >= _original_size) {
    throw std::out_of_range(
        "Index out of range for the compressed vector of size " +
        std::to_string(_original_size));
  }

  int hash = thirdai::hashing::MurmurHash(std::to_string(index).c_str(),
                                          std::to_string(index).length(),
                                          _seed_for_hashing) %
             _sketch_size;

  _indices[hash] = index;
  _values[hash] = value;
}

// ideally this method should not be called.
// what should we do with _original_size? I think we should let it remain the
// same but throw an error if assign is called on an uninitialized dragon vector

template <class T>
void DragonVector<T>::assign(uint32_t size, T value) {
  std::cout << "Warning: Assigning all indices are being set to 0. Also pass "
               "the index if want to set index to a specific value "
            << std::endl;

  if (_original_size == 0) {
    std::cout
        << ("Unsafe: Do not call assign on an unintialized compressed vector "
            "without specifying the value of "
            "original vector. Original vector size is being set to size");
  }

  _sketch_size = size;
  _original_size = size;
  _values.assign(_sketch_size, value);
  _indices.assign(_sketch_size, 0);
}

template <class T>
void DragonVector<T>::assign(uint32_t size, uint32_t index, T value,
                             uint32_t original_size) {
  _sketch_size = size;

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

  _values.assign(_sketch_size, value);
  _indices.assign(_sketch_size, index);
}

template <class T>
void DragonVector<T>::clear() {
  _sketch_size = 0;
  _original_size = 0;
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
  if (_sketch_size != vec._sketch_size) {
    throw std::length_error(
        "Cannot add two Dragon Sketches of different sizes");
  }

  if (_original_size != vec._original_size) {
    throw std::length_error(
        "Cannot add two Dragon Sketches with original vectors of different "
        "sizes");
  }

  std::vector<uint32_t> return_indices(_sketch_size, 0);
  std::vector<T> return_values(_sketch_size, 0);

#pragma omp parallel for default(none) shared( \
    vec, _sketch_size, _values, _indices, return_indices, return_values)

  for (uint32_t i = 0; i < _sketch_size; i++) {
    /*
     * s=s1+s2
     * If s1[index] is non-zero, we use value and index from s1, otherwise use
     * sketch 2 Should not use if-else because of branching overheads.
     */

    return_indices[i] = _indices[i] + (_indices[i] == 0) * vec._indices[i];
    return_values[i] = _values[i] + (_indices[i] == 0) * vec._values[i];
  }
  return DragonVector(return_indices, return_values, _sketch_size,
                      _original_size, _seed_for_hashing);
}

template <class T>
T DragonVector<T>::operator[](uint32_t index) const {
  return DragonVector<T>::get(index);
}

/*
 * Implementing utility methods for the class
 */
template <class T>
bool DragonVector<T>::isAllReducible() const {
  return false;
}

template <class T>
void DragonVector<T>::extend(const DragonVector<T>& vec) {
  // We should not check whether the seeds for hashing are the same for the two
  // Dragon vectors. We will directly append the indices and values of given
  // vector to the current one but leave all other parameters intact

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
  _sketch_size += vec._sketch_size;
  //_original_size remains the same
}

template <class T>
std::vector<DragonVector<T>> DragonVector<T>::split(
    size_t number_chunks) const {
  if (uint32_t(number_chunks) > _sketch_size) {
    std::cout
        << "Warning: The number of chunks to split the vector is more "
           "than the size of the Dragon vector. Some chunks will be empty";
  }

  std::vector<std::vector<uint32_t>> split_indices =
      thirdai::compression::SplitVector(_indices, number_chunks);
  std::vector<std::vector<T>> split_values =
      thirdai::compression::SplitVector(_values, number_chunks);

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
                                        split_indices[i].size(), _original_size,
                                        _seed_for_hashing));
  }
  return split_dragon;
}

template <class T>
std::vector<T> DragonVector<T>::decompressVector() const {
  std::vector<T> decompressedVector(_original_size, 0);
  for (uint32_t i = 0; i < _sketch_size; i++) {
    decompressedVector[_indices[i]] += _values[i];
  }
  return decompressedVector;
}

// concatenating is the same as extending for the time being
template <class T>
DragonVector<T>& DragonVector<T>::concat(const DragonVector<T>& vec) {
  extend(vec);
  return *this;
}
template class DragonVector<float>;

}  // namespace thirdai::compression
