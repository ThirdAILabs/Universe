#include "DragonVector.h"
#include <hashing/src/MurmurHash.h>
#include <_types/_uint32_t.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::compression {

template <class T>
DragonVector<T>::DragonVector(const std::vector<T>& vec,
                              float compression_density, int seed_for_hashing)
    : _sketch_size(std::max(compression_density * vec.size(),
                            std::min(vec.size(), _min_sketch_size))),
      _compression_density(compression_density),
      _seed_for_hashing(seed_for_hashing) {
  _indices.assign(_sketch_size, 0);
  _values.assign(_sketch_size, 0);

  float threshold = thirdai::compression::getThresholdForTopK(
      vec, _sketch_size, /*max_samples_for_random_sampling=*/100000);

  sketchVector(vec, threshold);
}

template <class T>
DragonVector<T>::DragonVector(std::vector<uint32_t> indices,
                              std::vector<T> values, uint32_t size,
                              int seed_for_hashing)
    : _sketch_size(size),
      _values(std::move(values)),
      _indices(std::move(indices)),
      _seed_for_hashing(seed_for_hashing) {}

template <class T>
void DragonVector<T>::sketchVector(const std::vector<T>& vec, float threshold) {
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

/*
 * Implementing std::vector's standard methods for the class
 */

template <class T>
T DragonVector<T>::get(uint32_t index) const {
  if (_sketch_size == 0) {
    throw std::logic_error(
        "Accessing elements from an empty compressed vector");
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

  int hash = thirdai::hashing::MurmurHash(std::to_string(index).c_str(),
                                          std::to_string(index).length(),
                                          _seed_for_hashing) %
             _sketch_size;

  _indices[hash] = index;
  _values[hash] = value;
}

// ideally this method should not be called.
template <class T>
void DragonVector<T>::assign(uint32_t size, T value) {
  std::cout << "Warning: Assigning all indices are being set to 0. Also pass "
               "the index if want to set index to a specific value "
            << std::endl;
  _sketch_size = size;
  _values.assign(_sketch_size, value);
  _indices.assign(_sketch_size, 0);
}

template <class T>
void DragonVector<T>::assign(uint32_t size, uint32_t index, T value) {
  _sketch_size = size;
  _values.assign(_sketch_size, value);
  _indices.assign(_sketch_size, index);
}

template <class T>
void DragonVector<T>::clear() {
  _sketch_size = 0;
  _values.clear();
  _indices.clear();
}

/*
 * Implementing Operator methods for the class
 */

template <class T>
DragonVector<T> DragonVector<T>::operator+(const DragonVector<T>& vec) {
  if (_seed_for_hashing != vec._seed_for_hashing) {
    throw std::invalid_argument(
        "Seeds for hashing of the two Dragon Sketches are different. Try "
        "concatenating the sketches");
  }
  if (_sketch_size != vec._sketch_size) {
    throw std::length_error(
        "Cannot add two Dragon Sketches of different sizes");
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
    return_values[i] == _values[i] + (_indices[i] == 0) * vec._values[i];
  }
  return DragonVector(return_indices, return_values, _sketch_size,
                      _seed_for_hashing);
}

template <class T>
T DragonVector<T>::operator[](uint32_t index) {
  return DragonVector<T>::get(index);
}

// methods for the Dragon vector class

template <class T>
bool DragonVector<T>::isAllReducible() const {
  return false;
}

template <class T>
void DragonVector<T>::extend(const DragonVector<T>& vec) {
  // we should not check whether the seeds for hashing are the same for the two
  // Dragon vectors we will directly append the indices and values of given
  // vector to the current one but leave all other parameters intact

  _indices.insert(std::end(_indices), std::begin(vec._indices),
                  std::end(vec._indices));
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
  _sketch_size += vec._sketch_size;
}

template <class T>
std::vector<DragonVector<T>> DragonVector<T>::split(int number_chunks) const {
  if (number_chunks > _sketch_size) {
    std::cout
        << "Warning: The number of chunks to split the vector is more "
           "than the size of the Dragon vector. Some chunks will be empty";
  }

  std::vector<std::vector<uint32_t>> split_indices =
      thirdai::compression::SplitVector(_indices, number_chunks);
  std::vector<std::vector<T>> split_values =
      thirdai::compression::SplitVector(_values, number_chunks);

  std::vector<DragonVector<T>> split_dragon;

  if (int(split_indices.size()) != number_chunks) {
    throw std::length_error(
        "Number of vectors received after splitting is not the same as the "
        "number of chunks");
  }

  for (size_t i = 0; i < split_indices.size(); i++) {
    if (split_indices[i].size() != split_values[i].size()) {
      throw std::length_error(
          "Size of indices and values array are not the same");
    }
    split_dragon.push_back(DragonVector(split_indices[i], split_values[i],
                                        split_indices[i].size(),
                                        _seed_for_hashing));
  }
  return split_dragon;
}

// concatenating is the same as extending for the time being
template <class T>
DragonVector<T> DragonVector<T>::concat(const DragonVector<T>& vec) {
  extend(vec);
  return this;
}

}  // namespace thirdai::compression
