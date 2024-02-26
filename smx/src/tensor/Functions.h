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
TensorPtr relu(const TensorPtr& in);

TensorPtr reluGrad(const TensorPtr& out, const TensorPtr& out_grad);

TensorPtr tanh(const TensorPtr& in);

TensorPtr tanhGrad(const TensorPtr& out, const TensorPtr& out_grad);

TensorPtr sigmoid(const TensorPtr& in);

TensorPtr sigmoidGrad(const TensorPtr& out, const TensorPtr& out_grad);

TensorPtr softmax(const TensorPtr& in);

TensorPtr softmaxGrad(const TensorPtr& out, const TensorPtr& out_grad);

/**
 * Linear algebra functions.
 */

TensorPtr add(const TensorPtr& a, const TensorPtr& b);

DenseTensorPtr linear(const TensorPtr& x, const DenseTensorPtr& w,
                      const DenseTensorPtr& b);

std::tuple<TensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const TensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
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
                         const DenseTensorPtr& embeddings,
                         const DenseTensorPtr& bias);

std::pair<DenseTensorPtr, DenseTensorPtr> embeddingGrad(
    const CsrTensorPtr& indices, const DenseTensorPtr& out_grad, bool bias);

/**
 * Loss functions.
 */

std::pair<DenseTensorPtr, TensorPtr> sparseCrossEntropy(
    const TensorPtr& logits, const TensorPtr& labels);

TensorPtr sparseCrossEntropyGrad(const TensorPtr& y, const TensorPtr& labels);

std::pair<DenseTensorPtr, TensorPtr> sparseBinaryCrossEntropy(
    const TensorPtr& logits, const TensorPtr& labels);

TensorPtr sparseBinaryCrossEntropyGrad(const TensorPtr& y,
                                       const TensorPtr& labels);

}  // namespace thirdai::smx