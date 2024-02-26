#pragma once

#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

class MaskedTensor final : public Tensor {
 public:
  MaskedTensor(DenseTensorPtr tensor, std::vector<bool> mask)
      : Tensor(tensor->shape(), tensor->dtype()),
        _tensor(std::move(tensor)),
        _mask(std::move(mask)) {
    CHECK(tensor->ndim() == 2, "Masked tensor must be 2d.");
    CHECK(tensor->shape(0) == mask.size(), "Mask size must match tensor.");
  }

  bool isSparse() const final { return false; }

 private:
  DenseTensorPtr _tensor;
  std::vector<bool> _mask;
};

using MaskedTensorPtr = std::shared_ptr<MaskedTensor>;

inline MaskedTensorPtr masked(const TensorPtr& tensor) {
  CHECK(tensor, "Tensor should not be null.");
  if (auto ptr = std::dynamic_pointer_cast<MaskedTensor>(tensor)) {
    return ptr;
  }
  return nullptr;
}

}  // namespace thirdai::smx