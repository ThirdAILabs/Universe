#pragma once

#include <new_dataset/src/featurization_pipeline/Column.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

namespace thirdai::data::columns {

// py::array::forcecast is safe because we want everything in terms of
// uint32_t/float.
template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
static void checkArrayis1D(const NumpyArray<T>& array);

template <typename T>
static void checkArrayIs2D(const NumpyArray<T>& array);

static void verifySparseArrayIndices(const NumpyArray<uint32_t>& array,
                                     uint32_t dim);
template <typename T>
static const T& getItemHelper(const py::buffer_info& buffer, uint64_t n);

template <typename T>
static typename ArrayColumn<T>::RowReference getRowHelper(
    const py::buffer_info& buffer, uint64_t n);

class PyTokenColumn final : public TokenColumn {
 public:
  PyTokenColumn(const NumpyArray<uint32_t>& array, std::optional<uint32_t> dim)
      : _dim(dim) {
    checkArrayis1D(array);

    if (dim) {
      verifySparseArrayIndices(array, *dim);
    }

    _buffer_info = array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  std::optional<DimensionInfo> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  const uint32_t& operator[](uint64_t n) const final {
    return getItemHelper<uint32_t>(_buffer_info, n);
  }

 private:
  py::buffer_info _buffer_info;
  std::optional<uint32_t> _dim;
};

class PyDenseFeatureColumn final : public DenseFeatureColumn {
 public:
  explicit PyDenseFeatureColumn(const NumpyArray<float>& array) {
    checkArrayis1D(array);

    _buffer_info = array.request();
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  std::optional<DimensionInfo> dimension() const final {
    return {{/* dim= */ 1, /* is_dense= */ true}};
  }

  const float& operator[](uint64_t n) const final {
    return getItemHelper<float>(_buffer_info, n);
  }

 private:
  py::buffer_info _buffer_info;
};

class PyTokenArrayColumn final : public TokenArrayColumn {
 public:
  PyTokenArrayColumn(const NumpyArray<uint32_t>& array,
                     std::optional<uint32_t> dim)
      : _dim(dim) {
    checkArrayIs2D(array);

    if (dim) {
      verifySparseArrayIndices(array, *dim);
    }

    _buffer_info = array.request();
  }

  std::optional<DimensionInfo> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<uint32_t>::RowReference operator[](
      uint64_t n) const final {
    return getRowHelper<uint32_t>(_buffer_info, n);
  }

 private:
  py::buffer_info _buffer_info;
  std::optional<uint32_t> _dim;
};

class PyDenseArrayColumn final : public DenseArrayColumn {
 public:
  explicit PyDenseArrayColumn(const NumpyArray<float>& array) {
    checkArrayIs2D(array);

    _buffer_info = array.request();
  }

  std::optional<DimensionInfo> dimension() const final {
    uint32_t dim = _buffer_info.shape[1];
    return {{dim, /* is_dense= */ true}};
  }

  uint64_t numRows() const final { return _buffer_info.shape[0]; }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<float>::RowReference operator[](uint64_t n) const final {
    return getRowHelper<float>(_buffer_info, n);
  }

 private:
  py::buffer_info _buffer_info;
};

template <typename T>
static void checkArrayis1D(const NumpyArray<T>& array) {
  if (array.ndim() != 1 && (array.ndim() != 2 || array.shape(1) != 1)) {
    throw std::invalid_argument(
        "Can only construct NumpyValueColumn from 1D numpy array.");
  }
}

template <typename T>
static void checkArrayIs2D(const NumpyArray<T>& array) {
  if (array.ndim() != 2) {
    throw std::invalid_argument(
        "Can only construct NumpyArrayColumn from 2D numpy array.");
  }
}

static void verifySparseArrayIndices(const NumpyArray<uint32_t>& array,
                                     uint32_t dim) {
  const uint32_t* data = array.data();
  for (uint32_t i = 0; i < array.size(); i++) {
    if (data[i] >= dim) {
      throw std::out_of_range("Cannot have index " + std::to_string(data[i]) +
                              " in Sparse Numpy Column of dimension " +
                              std::to_string(dim) + ".");
    }
  }
}

template <typename T>
static const T& getItemHelper(const py::buffer_info& buffer, uint64_t n) {
  return static_cast<const T*>(buffer.ptr)[n];
}

template <typename T>
static typename ArrayColumn<T>::RowReference getRowHelper(
    const py::buffer_info& buffer, uint64_t n) {
  uint64_t len = buffer.shape[1];
  const T* ptr = static_cast<const T*>(buffer.ptr) + len * n;
  return {ptr, len};
}

}  // namespace thirdai::data::columns