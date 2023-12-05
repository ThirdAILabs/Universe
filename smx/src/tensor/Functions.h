#pragma once

#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

TensorPtr transpose(const TensorPtr& tensor, const std::vector<size_t>& perm);

TensorPtr reshape(const TensorPtr& tensor, const Shape& new_shape);

}  // namespace thirdai::smx