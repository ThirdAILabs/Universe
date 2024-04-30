#pragma once

#include <smx/src/tensor/DenseTensor.h>

namespace thirdai::smx {

class MaskedTensor final : public DenseTensor {
 public:
  MaskedTensor(const Shape& shape, Dtype dtype)
      : DenseTensor(shape, dtype), _mask(shape[0], false) {
    CHECK(ndim() == 2, "Masked tensor must be 2d.");
  }

  static auto make(const Shape& shape, Dtype dtype) {
    return std::make_shared<MaskedTensor>(shape, dtype);
  }

  auto& mask() { return _mask; }

  const auto& mask() const { return _mask; }

  bool isMasked() const final { return true; }

 private:
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