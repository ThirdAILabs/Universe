#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <stdexcept>

namespace thirdai::smx {

TensorPtr reshape(const TensorPtr& tensor, const Shape& new_shape) {
  if (!tensor->shape().canReshapeTo(new_shape)) {
    throw std::invalid_argument("Cannot reshape tensor with shape " +
                                tensor->shape().toString() + " to shape " +
                                new_shape.toString() + ".");
  }

  if (tensor->isSparse()) {
    throw std::invalid_argument(
        "Reshaping sparse tensors is not yet supported.");
  }

  auto dense_tensor = dense(tensor);

  return DenseTensor::make(new_shape, dense_tensor->dtype(),
                           dense_tensor->handle());
}

}  // namespace thirdai::smx