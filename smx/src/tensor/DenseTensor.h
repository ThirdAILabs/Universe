#pragma once

#include <Eigen/Core>
#include <Eigen/src/Core/Array.h>
#include <Eigen/src/Core/Map.h>
#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/unsupported/Eigen/CXX11/Tensor>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/MemoryHandle.h>
#include <smx/src/tensor/Tensor.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::smx {

template <typename T, size_t NDim>
using EigenTensor = Eigen::TensorMap<Eigen::Tensor<T, NDim, Eigen::RowMajor>>;

template <typename T>
using EigenMatrix = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using EigenVector =
    Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic, Eigen::RowMajor>>;

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

  DenseTensor(const Shape& shape, Dtype dtype, MemoryHandlePtr data)
      : Tensor(shape, dtype),
        _strides(contiguousStrides(shape)),
        _data(std::move(data)) {
    _ptr = _data->ptr();

    if (_data->nbytes() != (shape.size() * sizeofDtype(dtype))) {
      throw std::invalid_argument(
          "Size of data does not match size of shape and dtype.");
    }
  }

  static auto make(const Shape& shape, Dtype dtype) {
    return std::make_shared<DenseTensor>(shape, dtype);
  }

  static auto make(const Shape& shape, Dtype dtype, MemoryHandlePtr data) {
    return std::make_shared<DenseTensor>(shape, dtype, data);
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

    std::array<size_t, NDim> dims;
    for (size_t i = 0; i < NDim; i++) {
      dims[i] = shapeAt(i);
    }

    return {data<T>(), dims};
  }

  template <typename T, size_t NDim>
  EigenTensor<const T, NDim> toEigen() const {
    if (NDim != ndim()) {
      throw std::invalid_argument("Cannot construct Eigen reference with " +
                                  std::to_string(NDim) +
                                  " dimensions to tensor with " +
                                  std::to_string(ndim()) + " dimensions.");
    }

    checkDtypeCompatability<T>();

    std::array<size_t, NDim> dims;
    for (size_t i = 0; i < NDim; i++) {
      dims[i] = shapeAt(i);
    }

    return {data<T>(), dims};
  }

  template <typename T>
  EigenMatrix<T> eigenMatrix() {
    checkDtypeCompatability<T>();

    if (ndim() == 2) {
      return {data<T>(), static_cast<int64_t>(shapeAt(0)),
              static_cast<int64_t>(shapeAt(1))};
    }

    size_t last_dim = shapeAt(ndim() - 1);
    return {data<T>(), static_cast<int64_t>(shape().size() / last_dim),
            static_cast<int64_t>(last_dim)};
  }

  template <typename T>
  EigenMatrix<const T> eigenMatrix() const {
    checkDtypeCompatability<T>();

    if (ndim() == 2) {
      return {data<T>(), static_cast<int64_t>(shapeAt(0)),
              static_cast<int64_t>(shapeAt(1))};
    }

    size_t last_dim = shapeAt(ndim() - 1);
    return {data<T>(), static_cast<int64_t>(shape().size() / last_dim),
            static_cast<int64_t>(last_dim)};
  }

  template <typename T>
  EigenVector<T> eigenVector() {
    if (ndim() != 1) {
      throw std::invalid_argument("Can only convert 1D tensor to vector.");
    }
    return {data<T>(), static_cast<int64_t>(shapeAt(0))};
  }

  template <typename T>
  EigenVector<const T> eigenVector() const {
    if (ndim() != 1) {
      throw std::invalid_argument("Can only convert 1D tensor to vector.");
    }
    return {data<T>(), static_cast<int64_t>(shapeAt(0))};
  }

  template <typename T>
  EigenArray<T> eigenArray() {
    checkDtypeCompatability<T>();

    return {data<T>(), static_cast<int64_t>(_shape.size())};
  }

  template <typename T>
  EigenArray<const T> eigenArray() const {
    checkDtypeCompatability<T>();

    return {data<T>(), static_cast<int64_t>(_shape.size())};
  }

  template <typename T>
  const T* data() const {
    return static_cast<const T*>(_ptr);
  }

  template <typename T>
  T* data() {
    return static_cast<T*>(_ptr);
  }

  const MemoryHandlePtr& handle() const { return _data; }

 private:
  template <typename T>
  void checkDtypeCompatability() const {
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

using DenseTensorPtr = std::shared_ptr<DenseTensor>;

inline DenseTensorPtr asDense(const TensorPtr& tensor) {
  if (auto ptr = std::dynamic_pointer_cast<DenseTensor>(tensor)) {
    return ptr;
  }
  throw std::invalid_argument("Cannot convert sparse tensor to dense tensor.");
}

}  // namespace thirdai::smx