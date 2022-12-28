#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::computation_graph {

class ActivationsManager {
 public:
  explicit ActivationsManager(
      std::vector<tensor::ActivationTensorPtr> activation_tensors)
      : _activation_tensors(std::move(activation_tensors)),
        _allocated_batch_size(0),
        _using_sparsity(true) {}

  void reallocateForBatch(uint32_t batch_size, bool use_sparsity) {
    if (batch_size <= _allocated_batch_size &&
        use_sparsity == _using_sparsity) {
      return;
    }

    for (auto& tensor : _activation_tensors) {
      tensor->allocate(batch_size, use_sparsity);
    }
  }

  const std::vector<tensor::ActivationTensorPtr>& activationTensors() const {
    return _activation_tensors;
  }

  uint32_t currentBatchSize() const { return _allocated_batch_size; }

 private:
  std::vector<tensor::ActivationTensorPtr> _activation_tensors;

  uint32_t _allocated_batch_size;
  bool _using_sparsity;
};

}  // namespace thirdai::bolt::nn::computation_graph