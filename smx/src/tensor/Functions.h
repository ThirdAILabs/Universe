#pragma once

#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Tensor.h>
#include <smx/src/tensor/functions/sparse/NeuronIndex.h>

namespace thirdai::smx {

/**
 * Tensor shape functions.
 */
TensorPtr transpose(const TensorPtr& tensor, const std::vector<size_t>& perm);

TensorPtr reshape(const TensorPtr& tensor, const Shape& new_shape);

/**
 * Activation functions.
 */
DenseTensorPtr relu(const TensorPtr& in);

DenseTensorPtr reluGrad(const TensorPtr& out, const TensorPtr& out_grad);

DenseTensorPtr tanh(const TensorPtr& in);

DenseTensorPtr tanhGrad(const TensorPtr& out, const TensorPtr& out_grad);

DenseTensorPtr sigmoid(const TensorPtr& in);

DenseTensorPtr sigmoidGrad(const TensorPtr& out, const TensorPtr& out_grad);

DenseTensorPtr softmax(const TensorPtr& in);

DenseTensorPtr softmaxGrad(const TensorPtr& out, const TensorPtr& out_grad);

/**
 * Linear algebra functions.
 */

TensorPtr add(const TensorPtr& a, const TensorPtr& b);

DenseTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                      const DenseTensorPtr& b);

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const DenseTensorPtr& y_grad, bool compute_x_grad);

CsrTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                    const DenseTensorPtr& b, float sparsity,
                    const NeuronIndexPtr& neuron_index,
                    const TensorPtr& labels);

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const CsrTensorPtr& y_grad);

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