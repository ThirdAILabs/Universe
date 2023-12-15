#include "DenseTensor.h"

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

}  // namespace thirdai::smx