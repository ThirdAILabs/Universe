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
#include <string>

namespace thirdai::smx {

template <typename T, size_t NDim>
using EigenTensor = Eigen::Map<Eigen::Tensor<T, NDim, Eigen::RowMajor>>;

template <typename T>
using EigenArray =
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

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

  template <typename T, size_t NDim>
  EigenTensor<T, NDim> toEigen() {
    if (NDim != ndim()) {
      throw std::invalid_argument("Cannot construct Eigen reference with " +
                                  std::to_string(NDim) +
                                  " dimensions to tensor with " +
                                  std::to_string(ndim()) + " dimensions.");
    }

    checkDtypeCompatability<T>();

    return {_ptr, _shape.vector()};
  }

  template <typename T, size_t NDim>
  EigenTensor<T, NDim> reshapeToEigen(const Shape& shape) {
    if (NDim != shape.ndim()) {
      throw std::invalid_argument(
          "Requesting to reshape to eigen tensor with " + std::to_string(NDim) +
          " dimensions, but provided shape with " +
          std::to_string(shape.ndim()) + " dimensions.");
    }

    if (!_shape.canReshapeTo(shape)) {
      throw std::invalid_argument("Cannot reshape tensor with shape " +
                                  _shape.toString() + " to shape " +
                                  shape.toString() + ".");
    }

    checkDtypeCompatability<T>();

    return {_ptr, shape.vector()};
  }

  template <typename T>
  EigenArray<T> eigenArray() {
    checkDtypeCompatability<T>();

    return {_ptr, _shape.size()};
  }

  template <typename T>
  T* data() {
    return _ptr;
  }

  template <typename T>
  const T* data() const {
    return _ptr;
  }

 private:
  template <typename T>
  void checkDtypeCompatability() {
    if (getDtype<T>() != _dtype) {
      throw std::invalid_argument("Canot convert tensor of type " +
                                  toString(_dtype) + " to type " +
                                  toString(getDtype<T>()) + ".");
    }
  }

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