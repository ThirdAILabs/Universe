#pragma once

#include "Tensor.h"

namespace thirdai::bolt::nn::tensor {

class ActivationTensor final : public Tensor {
 public:
  ActivationTensor(uint32_t dim, uint32_t num_nonzeros);

  BoltVector& getVector(uint32_t index) final;

  void allocate(uint32_t batch_size, bool use_sparsity);

  void updateSparsity(uint32_t num_nonzeros);

 private:
  std::vector<BoltVector> _vectors;
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