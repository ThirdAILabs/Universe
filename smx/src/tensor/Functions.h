#pragma once

#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>

namespace thirdai::smx {

/**
 * Tensor shape functions.
 */
TensorPtr transpose(const TensorPtr& tensor, const std::vector<size_t>& perm);

TensorPtr reshape(const TensorPtr& tensor, const Shape& new_shape);

/**
 * Activation functions.
 */
DenseTensorPtr relu(const DenseTensorPtr& in);

DenseTensorPtr reluGrad(const DenseTensorPtr& out,
                        const DenseTensorPtr& out_grad);

DenseTensorPtr tanh(const DenseTensorPtr& in);

DenseTensorPtr tanhGrad(const DenseTensorPtr& out,
                        const DenseTensorPtr& out_grad);

DenseTensorPtr sigmoid(const DenseTensorPtr& in);

DenseTensorPtr sigmoidGrad(const DenseTensorPtr& out,
                           const DenseTensorPtr& out_grad);

DenseTensorPtr softmax(const DenseTensorPtr& in);

DenseTensorPtr softmaxGrad(const DenseTensorPtr& out,
                           const DenseTensorPtr& out_grad);

/**
 * Linear algebra functions
 */

TensorPtr add(const TensorPtr& a, const TensorPtr& b);

TensorPtr linear(const TensorPtr& x, const DenseTensorPtr& w,
                 const DenseTensorPtr& b);

std::tuple<TensorPtr, TensorPtr, TensorPtr> linearGrad(const TensorPtr& x,
                                                       const DenseTensorPtr& w,
                                                       const DenseTensorPtr& b,
                                                       const TensorPtr& y_grad,
                                                       bool compute_x_grad);

}  // namespace thirdai::smx