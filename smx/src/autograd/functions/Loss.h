#pragma once

#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

VariablePtr crossEntropy(const VariablePtr& logits, const TensorPtr& labels);

VariablePtr binaryCrossEntropy(const VariablePtr& logits,
                               const TensorPtr& labels);

}  // namespace thirdai::smx