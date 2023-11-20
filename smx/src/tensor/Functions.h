#pragma once

#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

TensorPtr transpose(const TensorPtr& tensor, std::vector<size_t> perm);

}  // namespace thirdai::smx