#pragma once

#include <_types/_uint32_t.h>
#include <data/src/columns/Column.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace py = pybind11;

namespace thirdai::data {

// py::array::forcecast is safe because we want everything in terms of
// uint32_t/float.
template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
class NumpyValueColumn : public ValueColumn<T> {
 public:
  size_t numRows() const final { return _buffer_info.shape[0]; }

  std::optional<ColumnDimension> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  RowView<T> row(size_t n) const final {
    const T* ptr = static_cast<const T*>(_buffer_info.ptr) + n;
    return {ptr, 1};
  }

  const T& value(size_t n) const final {
    return static_cast<const T*>(_buffer_info.ptr)[n];
  }

  void shuffle(const std::vector<size_t>& permutation) final {
    (void)permutation;
    throw std::runtime_error("Shuffling is not supported for numpy columns.");
  }

  ColumnPtr concat(ColumnPtr&& other) final {
    (void)other;
    throw std::runtime_error("Concat is not supported for numpy columns.");
  }

 protected:
  py::buffer_info _buffer_info;
  std::optional<size_t> _dim;
};

class NumpyTokenColumn final : public NumpyValueColumn<uint32_t> {
 public:
  NumpyTokenColumn(const NumpyArray<uint32_t>& array,
                   std::optional<size_t> dim);
};

class NumpyDecimalColumn final : public NumpyValueColumn<float> {
 public:
  explicit NumpyDecimalColumn(const NumpyArray<uint32_t>& array);
};

template <typename T>
class NumpyArrayColumn : public ArrayColumn<T> {
 public:
  std::optional<ColumnDimension> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  size_t numRows() const final { return _buffer_info.shape[0]; }

  RowView<T> row(size_t n) const final {
    uint64_t len = _buffer_info.shape[1];
    const T* ptr = static_cast<const T*>(_buffer_info.ptr) + len * n;
    return {ptr, len};
  }

  void shuffle(const std::vector<size_t>& permutation) final {
    (void)permutation;
    throw std::runtime_error("Shuffling is not supported for numpy columns.");
  }

  ColumnPtr concat(ColumnPtr&& other) final {
    (void)other;
    throw std::runtime_error("Concat is not supported for numpy columns.");
  }

 protected:
  py::buffer_info _buffer_info;
  std::optional<size_t> _dim;
};

class NumpyTokenArrayColumn final : public NumpyValueColumn<uint32_t> {
 public:
  NumpyTokenArrayColumn(const NumpyArray<uint32_t>& array,
                        std::optional<size_t> dim);
};

class NumpyDecimalArrayColumn final : public NumpyValueColumn<float> {
 public:
  explicit NumpyDecimalArrayColumn(const NumpyArray<uint32_t>& array);
};

}  // namespace thirdai::data