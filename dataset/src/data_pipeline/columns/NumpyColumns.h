#pragma once

#include <dataset/src/data_pipeline/Column.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

namespace thirdai::dataset {

// py::array::forcecast is safe because we want everything in terms of
// uint32_t/float.
template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
class NumpyValueColumn final : public ValueColumn<T> {
  static_assert(std::is_same<T, uint32_t>::value ||
                    std::is_same<T, float>::value,
                "Only numpy arrays of type uint32 or float32 can be used to "
                "construct columns.");

 public:
  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit NumpyValueColumn(const NumpyArray<uint32_t>& array, uint32_t dim)
      : _dim(dim) {
    checkArrayis1D(array);

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

  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit NumpyValueColumn(const NumpyArray<float>& array) : _dim(1) {
    checkArrayis1D(array);

    _buffer_info = array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  std::optional<DimensionInfo> dimension() const final {
    if constexpr (std::is_same<T, uint32_t>::value ||
                  std::is_same<T, float>::value) {
      return {{_dim, std::is_same<T, float>::value}};
    }
    return std::nullopt;
  }

  const T& operator[](uint64_t n) const final {
    return static_cast<const T*>(_buffer_info.ptr)[n];
  }

 private:
  static void checkArrayis1D(const NumpyArray<uint32_t>& array) {
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
                    std::is_same<T, float>::value,
                "Only numpy arrays of type uint32 or float32 can be used to "
                "construct columns.");

 public:
  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, uint32_t>::value, bool> = true>
  explicit NumpyArrayColumn(const NumpyArray<uint32_t>& array, uint32_t dim)
      : _dim(dim) {
    checkArrayIs2D(array);

    _buffer_info = array.request();
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

  // This uses SFINAE to disable the folowing constructor if T is not a certain
  // type. https://en.cppreference.com/w/cpp/types/enable_if
  template <typename U = T,
            std::enable_if_t<std::is_same<U, float>::value, bool> = true>
  explicit NumpyArrayColumn(const NumpyArray<float>& array) {
    checkArrayIs2D(array);

    _buffer_info = array.request();
    _dim = _buffer_info.shape[1];
  }

  std::optional<DimensionInfo> dimension() const final {
    if constexpr (std::is_same<T, uint32_t>::value ||
                  std::is_same<T, float>::value) {
      return {{_dim, std::is_same<T, float>::value}};
    }
    return std::nullopt;
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<T>::RowReference operator[](uint64_t n) const final {
    uint64_t len = _buffer_info.shape[1];
    const T* ptr = static_cast<const T*>(_buffer_info.ptr) + len * n;

    return {ptr, len};
  }

 private:
  static void checkArrayIs2D(const NumpyArray<uint32_t>& array) {
    if (array.ndim() != 2) {
      throw std::invalid_argument(
          "Can only construct NumpyArrayColumn from 2D numpy array.");
    }
  }

  py::buffer_info _buffer_info;
  uint32_t _dim;
};

}  // namespace thirdai::dataset