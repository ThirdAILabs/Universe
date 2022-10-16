#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::dataset {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
class NumpyValueColumn final : public ValueColumn<T> {
 public:
  explicit NumpyValueColumn(const NumpyArray<T>& numpy_array) {
    if (numpy_array.ndim() != 1) {
      throw std::invalid_argument(
          "Can only construct NumpyValueColumn from 1D numpy array.");
    }

    _buffer_info = numpy_array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  const T& operator[](size_t n) const final {
    return static_cast<const T*>(_buffer_info.ptr)[n];
  }

 private:
  py::buffer_info _buffer_info;
};

template <typename T>
class NumpyArrayColumn final : public ArrayColumn<T> {
 public:
  explicit NumpyArrayColumn(const NumpyArray<T>& numpy_array) {
    if (numpy_array.ndim() != 2) {
      throw std::invalid_argument(
          "Can only construct NumpyArrayColumn from 2D numpy array.");
    }

    _buffer_info = numpy_array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  typename ArrayColumn<T>::RowReference operator[](size_t n) const final {
    uint64_t len = _buffer_info.shape[1];
    const T* ptr = static_cast<const T*>(_buffer_info.ptr) + len * n;

    return {ptr, len};
  }

 private:
  py::buffer_info _buffer_info;
};

}  // namespace thirdai::dataset