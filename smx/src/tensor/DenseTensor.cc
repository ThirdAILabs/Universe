#include "DenseTensor.h"
#include <smx/src/tensor/Dtype.h>
#include <string>

namespace thirdai::smx {

DenseTensor::DenseTensor(const Shape& shape, Dtype dtype)
    : Tensor(shape, dtype),
      _strides(contiguousStrides(shape)),
      _data(DefaultMemoryHandle::allocate(shape.size() * sizeofDtype(dtype))) {
  _ptr = _data->ptr();
}

DenseTensor::DenseTensor(const Shape& shape, Dtype dtype, MemoryHandlePtr data)
    : Tensor(shape, dtype),
      _strides(contiguousStrides(shape)),
      _data(std::move(data)) {
  _ptr = _data->ptr();

  if (_data->nbytes() != (shape.size() * sizeofDtype(dtype))) {
    throw std::invalid_argument(
        "Size of data does not match size of shape and dtype.");
  }
}

DenseTensorPtr DenseTensor::make(const std::vector<float>& data,
                                 const Shape& shape) {
  CHECK(shape.size() == data.size(),
        "Cannot construct tensor with shape " + shape.toString() +
            " from data of size " + std::to_string(data.size()) + ".");
  auto tensor = make(shape, Dtype::f32);
  std::copy(data.begin(), data.end(), tensor->data<float>());
  return tensor;
}

DenseTensorPtr DenseTensor::make(const std::vector<uint32_t>& data,
                                 const Shape& shape) {
  CHECK(shape.size() == data.size(),
        "Cannot construct tensor with shape " + shape.toString() +
            " from data of size " + std::to_string(data.size()) + ".");
  auto tensor = make(shape, Dtype::u32);
  std::copy(data.begin(), data.end(), tensor->data<uint32_t>());
  return tensor;
}

std::shared_ptr<DenseTensor> DenseTensor::index(
    const std::vector<size_t>& indices) {
  CHECK(indices.size() <= ndim(),
        "Cannot index a tensor with ndim=" + std::to_string(ndim()) + " with " +
            std::to_string(indices.size()) + " indices.");

  size_t offset = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    CHECK(indices[i] < _shape[i], "Cannot index the " + std::to_string(i) +
                                      "-th dimension of tensor with shape " +
                                      _shape.toString() + " with index " +
                                      std::to_string(indices[i]) + ".");
    offset += indices[i] * _strides[i];
  }
  offset *= sizeofDtype(_dtype);

  return std::shared_ptr<DenseTensor>(new DenseTensor(
      _shape.slice(indices.size()), _strides.slice(indices.size()), _dtype,
      reinterpret_cast<uint8_t*>(_ptr) + offset, _data));
}

}  // namespace thirdai::smx