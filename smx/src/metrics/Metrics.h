#pragma once

#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

/**
 * Retrieval Metrics
 */

float precision(const TensorPtr& scores, const CsrTensorPtr& labels, size_t k);

float recall(const TensorPtr& scores, const CsrTensorPtr& labels, size_t k);

}  // namespace thirdai::smx