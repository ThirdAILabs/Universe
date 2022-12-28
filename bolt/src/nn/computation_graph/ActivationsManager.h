#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::computation_graph {

class ActivationsManager {
 public:
  explicit ActivationsManager(
      std::vector<tensor::ActivationTensorPtr> activation_tensors);

  void reallocateForBatch(uint32_t batch_size, bool use_sparsity);

  const std::vector<tensor::ActivationTensorPtr>& activationTensors() const;

  void resetOutputGradients(uint32_t index_in_batch);

  uint32_t currentBatchSize() const;

 private:
  std::vector<tensor::ActivationTensorPtr> _activation_tensors;

  uint32_t _allocated_batch_size;
  bool _using_sparsity;
};

}  // namespace thirdai::bolt::nn::computation_graph