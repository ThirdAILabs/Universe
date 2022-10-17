#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

namespace thirdai::dataset {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
class NumpyValueColumn final : public ValueColumn<T> {
  static_assert(std::is_same<T, uint32_t>::value ||
                std::is_same<T, float>::value);

 public:
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit NumpyValueColumn(const NumpyArray<uint32_t>& array, uint32_t dim)
      : _dim(dim) {
    checkArrayDim(array);

    _buffer_info = array.request();

    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      uint32_t index = operator[](row_index);
      if (index >= _dim) {
        throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                " in NumpyIntegerValueColumn of dimension " +
                                std::to_string(_dim) + ".");
      }
    }
  }

  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit NumpyValueColumn(const NumpyArray<float>& array) : _dim(1) {
    checkArrayDim(array);

    _buffer_info = array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  uint32_t dim() const final { return _dim; }

  const T& operator[](uint64_t n) const final {
    return static_cast<const T*>(_buffer_info.ptr)[n];
  }

 private:
  static void checkArrayDim(const NumpyArray<uint32_t>& array) {
    if (array.ndim() != 1 && (array.ndim() != 2 || array.shape(1) != 1)) {
      throw std::invalid_argument(
          "Can only construct NumpyValueColumn from 1D numpy array.");
    }
  }

  py::buffer_info _buffer_info;
  uint32_t _dim;
};

template <typename T>
class NumpyArrayColumn final : public ArrayColumn<T> {
  static_assert(std::is_same<T, uint32_t>::value ||
                std::is_same<T, float>::value);

 public:
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit NumpyArrayColumn(const NumpyArray<uint32_t>& array, uint32_t dim)
      : _dim(dim) {
    checkArrayDim(array);

    _buffer_info = array.request();
  }

  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit NumpyArrayColumn(const NumpyArray<float>& array) {
    checkArrayDim(array);

    _buffer_info = array.request();
    _dim = _buffer_info.shape[1];

    for (uint64_t row_index = 0; row_index < numRows(); row_index++) {
      for (uint32_t index : operator[](row_index)) {
        if (index >= _dim) {
          throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                  " in NumpyIntegerArrayColumn of dimension " +
                                  std::to_string(_dim) + ".");
        }
      }
    }
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  uint32_t dim() const final { return _dim; }

  typename ArrayColumn<T>::RowReference operator[](uint64_t n) const final {
    uint64_t len = _buffer_info.shape[1];
    const T* ptr = static_cast<const T*>(_buffer_info.ptr) + len * n;

    return {ptr, len};
  }

 private:
  static void checkArrayDim(const NumpyArray<uint32_t>& array) {
    if (array.ndim() != 2) {
      throw std::invalid_argument(
          "Can only construct NumpyArrayColumn from 2D numpy array.");
    }
  }

  py::buffer_info _buffer_info;
  uint32_t _dim;
};

}  // namespace thirdai::dataset