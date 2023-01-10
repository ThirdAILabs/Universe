#pragma once

#include "Tensor.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::tensor {

/**
 * Subclass of Tensor which represents the outputs of an op.
 */
class ActivationTensor final : public Tensor {
 public:
  ActivationTensor(uint32_t dim, ops::OpPtr source, TensorList inputs);

  static std::shared_ptr<ActivationTensor> make(uint32_t dim, ops::OpPtr source,
                                                TensorList inputs);

  /**
   * Returns the op which whose activations are stored in the tensor.
   */
  ops::OpPtr source() const;

  /**
   * Returns the inputs to the tensor.
   */
  const TensorList& inputs() const;

  std::optional<uint32_t> numNonzeros(bool use_sparsity) const final;

  BoltVector& getVector(uint32_t index) final;

  void forward(uint32_t index_in_batch, bool training);

  void backpropagate(uint32_t index_in_batch);

  /**
   * Reallocates the number of vectors stored in the Tensor to reflect either a
   * change in the batch size the model is processing or a change in whether
   * sparsity is being used for the forward pass or not. If use_sparsity is
   * false then it will allocate dense vectors if the dimension of the tensor.
   * Otherwise it will allocate sparse vectors with _sparse_nonzeros number of
   * nonzero elements.
   */
  void allocate(uint32_t batch_size, bool use_sparsity);

  /**
   * Returns the current shape of the active neurons, activations, and
   * gradients.
   */
  std::vector<uint32_t> shape() const;

  /**
   * Returns a non-owning ptr to the active neurons.
   */
  const uint32_t* activeNeuronsPtr() const;

  /**
   * Returns a non-owning ptr to the activations.
   */
  const float* activationsPtr() const;

  /**
   * Returns a non-owning ptr to the gradients.
   */
  const float* gradientsPtr() const;

 private:
  ops::OpPtr _source;
  TensorList _inputs;

  std::vector<BoltVector> _vectors;

  // Storing the activations and active neurons as a continuous array and taking
  // pointers into it allows us to map the activations to numpy arrays without
  // copying.
  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using ActivationTensorPtr = std::shared_ptr<ActivationTensor>;

ActivationTensorPtr asActivationTensor(const tensor::TensorPtr& tensor);

}  // namespace thirdai::bolt::nn::tensor