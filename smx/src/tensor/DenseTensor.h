#pragma once

#include <Eigen/Core>
#include <Eigen/src/Core/Array.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/unsupported/Eigen/CXX11/Tensor>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/MemoryHandle.h>
#include <smx/src/tensor/Tensor.h>
#include <stdexcept>

namespace thirdai::smx {

class DenseTensor final : public Tensor {
 public:
  DenseTensor(const Shape& shape, Dtype dtype)
      : Tensor(shape, dtype),
        _strides(contiguousStrides(shape)),
        _data(
            DefaultMemoryHandle::allocate(shape.size() * sizeofDtype(dtype))) {
    _ptr = _data->ptr();
  }

  const Shape& strides() const { return _strides; }

  bool isSparse() const final { return false; }

  bool isContiguous() const { return isContiguous(0, ndim()); }

  bool isContiguous(size_t start_dim, size_t end_dim) const {
    for (size_t i = start_dim; i < end_dim - 1; i++) {
      if (_strides[i] != _shape[i + 1] * _strides[i + 1]) {
        return false;
      }
    }
    return true;
  }

  template <typename T, size_t NDim>
  Eigen::Map<Eigen::Tensor<T, NDim, Eigen::RowMajor>> eigenTensor() {
    if (NDim != ndim()) {
      throw std::invalid_argument("Cannot construct Eigen reference with " +
                                  std::to_string(NDim) +
                                  " dimensions to tensor with " +
                                  std::to_string(ndim()) + " dimensions.");
    }

    if (getDtype<T>() != _dtype) {
      throw std::invalid_argument("Canot convert tensor of type " +
                                  toString(_dtype) + " to type " +
                                  toString(getDtype<T>()) + ".");
    }

    return {_ptr, _shape.vector()};
  }

  template <typename T>
  Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic, Eigen::RowMajor>> eigenArray() {
    if (getDtype<T>() != _dtype) {
      throw std::invalid_argument("Canot convert tensor of type " +
                                  toString(_dtype) + " to type " +
                                  toString(getDtype<T>()) + ".");
    }

    return {_ptr, _shape.size()};
  }

 private:
  static Shape contiguousStrides(const Shape& shape) {
    std::vector<size_t> strides(shape.ndim());
    size_t stride = 1;
    for (size_t i = 0; i < shape.ndim(); i++) {
      strides[shape.ndim() - i - 1] = stride;
      stride *= shape[shape.ndim() - i - 1];
    }
    return Shape(strides);
  }

  Shape _strides;

  void* _ptr;
  MemoryHandlePtr _data;
};

}  // namespace thirdai::smx