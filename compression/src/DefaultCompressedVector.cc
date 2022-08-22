#include "DefaultCompressedVector.h"
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
DefaultCompressedVector<T>::DefaultCompressedVector(const std::vector<T>& vec)
    : _sketch_size(vec.size()) {
  _values.clear();
  _values.insert(std::end(_values), std::begin(vec), std::end(vec));
}

template <class T>
DefaultCompressedVector<T>::DefaultCompressedVector(const T* values,
                                                    uint32_t size)
    : _sketch_size(size) {
  _values.assign(values, values + size);
}

template <class T>
DefaultCompressedVector<T>::DefaultCompressedVector(
    const DefaultCompressedVector<T>& vec)
    : CompressedVector<T>(vec), _sketch_size(vec._sketch_size) {
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
}

/*
 * Implementing std::vector's standard methods for the class
 */

template <class T>
T DefaultCompressedVector<T>::get(uint32_t index) const {
  return _values[index];
}

template <class T>
void DefaultCompressedVector<T>::set(uint32_t index, T value) {
  _values[index] = value;
}

template <class T>
void DefaultCompressedVector<T>::assign(uint32_t size, T value) {
  _values.assign(size, value);
  _sketch_size = size;
}

template <class T>
void DefaultCompressedVector<T>::clear() {
  _values.clear();
  _sketch_size = 0;
}

/*
 * Implementing Operator methods for the class
 */

template <class T>
DefaultCompressedVector<T> DefaultCompressedVector<T>::operator+(
    DefaultCompressedVector<T> const& vec) const {
  if (_sketch_size != vec._sketch_size) {
    throw std::length_error(
        "Cannot add Default Compressed Vectors of different sizes");
  }
  std::vector<T> return_values(_sketch_size, 0);
#pragma omp parallel for default(none) \
    shared(return_values, _sketch_size, _values, vec)
  for (uint32_t i = 0; i < _sketch_size; i++) {
    return_values[i] = _values[i] + vec._values[i];
  }
  return DefaultCompressedVector(return_values);
}

// normally we use vec[i]=value to set it. here we are just fetching the i'th
// value and not setting it
template <class T>
T DefaultCompressedVector<T>::operator[](uint32_t index) const {
  return get(index);
}

/*
 * Implementing utility methods for the class
 */

template <class T>
bool DefaultCompressedVector<T>::isAllReducible() const {
  return false;
}

template <class T>
void DefaultCompressedVector<T>::extend(const DefaultCompressedVector<T>& vec) {
  _values.insert(std::end(_values), std::begin(vec._values),
                 std::end(vec._values));
}

template <class T>
std::vector<DefaultCompressedVector<T>> DefaultCompressedVector<T>::split(
    size_t number_chunks) const {
  if (uint32_t(number_chunks) > _sketch_size) {
    std::cout
        << "Warning: The number of chunks to split the vector is more "
           "than the size of the Dragon vector. Some chunks will be empty";
  }
  std::vector<std::vector<T>> split_values =
      thirdai::compression::SplitVector(_values, number_chunks);

  std::vector<DefaultCompressedVector<T>> split_default;

  if (split_values.size() != number_chunks) {
    throw std::length_error(
        "Number of vectors received after splitting is not the same as the "
        "number of chunks");
  }

  for (size_t i = 0; i < split_values.size(); i++) {
    split_default.push_back(DefaultCompressedVector(split_values[i]));
  }
  return split_default;
}

template <class T>
DefaultCompressedVector<T>& DefaultCompressedVector<T>::concat(
    const DefaultCompressedVector<T>& vec) {
  extend(vec);
  return *this;
}
template <class T>
std::vector<T> DefaultCompressedVector<T>::decompressVector() const {
  return _values;
}

template class DefaultCompressedVector<float>;
}  // namespace thirdai::compression