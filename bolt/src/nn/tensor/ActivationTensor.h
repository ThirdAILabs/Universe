#pragma once

#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

/**
 * Subclass of Tensor which represents the outputs of an op.
 */
class ActivationTensor final : public Tensor {
 public:
  ActivationTensor(uint32_t dim, uint32_t sparse_nonzeros);

  static std::shared_ptr<ActivationTensor> make(uint32_t dim,
                                                uint32_t sparse_nonzeros);

  std::optional<uint32_t> numNonzeros() const final;

  BoltVector& getVector(uint32_t index) final;

  /**
   * Reallocates the number of vectors stored in the Tensor to reflect either a
   * change in the batch size the computation graph is processing or a change in
   * wether sparsity is being used for the forward pass or not. If use_sparsity
   * is false then it will allocate dense vectors if the dimension of the
   * tensor. Otherwise it will allocate sparse vectors with _sparse_nonzeros
   * number of nonzero elements.
   */
  void allocate(uint32_t batch_size, bool use_sparsity);

  /**
   * Updates the sparsity of the tensor by changing its number of
   * sparse_nonzeros.
   */
  void updateSparsity(uint32_t new_sparse_nonzeros);

 private:
  std::vector<BoltVector> _vectors;
  uint32_t _sparse_nonzeros;
  bool _using_sparsity;

  // Storing the activations and active neurons as a continuous array and taking
  // pointers into it allows us to map the activations to numpy arrays without
  // copying.
  std::vector<uint32_t> _active_neurons;
  std::vector<float> _activations;
  std::vector<float> _gradients;
};

using ActivationTensorPtr = std::shared_ptr<ActivationTensor>;

}  // namespace thirdai::bolt::nn::tensor