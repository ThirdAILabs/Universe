#pragma once

#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

VariablePtr transpose(const VariablePtr& input,
                      const std::vector<size_t>& perm);

VariablePtr reshape(const VariablePtr& input, const Shape& new_shape);

VariablePtr concat(const std::vector<VariablePtr>& inputs, size_t dim);

}  // namespace thirdai::smx