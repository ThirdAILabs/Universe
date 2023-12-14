#pragma once

#include <smx/src/tensor/CsrTensor.h>
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
 * Linear algebra functions.
 */

TensorPtr add(const TensorPtr& a, const TensorPtr& b);

DenseTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                      const DenseTensorPtr& b);

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const TensorPtr& y_grad, bool compute_x_grad);

/**
 * NN functions.
 */

DenseTensorPtr embedding(const CsrTensorPtr& indices,
                         const DenseTensorPtr& embeddings, bool reduce_mean);

DenseTensorPtr embeddingGrad(const CsrTensorPtr& indices,
                             const DenseTensorPtr& out_grad, bool reduce_mean);

/**
 * Loss functions.
 */

std::pair<DenseTensorPtr, DenseTensorPtr> sparseCrossEntropy(
    const DenseTensorPtr& logits, const DenseTensorPtr& labels);

DenseTensorPtr sparseCrossEntropyGrad(const DenseTensorPtr& y,
                                      const DenseTensorPtr& labels);

}  // namespace thirdai::smx