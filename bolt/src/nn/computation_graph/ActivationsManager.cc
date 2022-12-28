#include "ActivationsManager.h"

namespace thirdai::bolt::nn::computation_graph {

ActivationsManager::ActivationsManager(
    std::vector<tensor::ActivationTensorPtr> activation_tensors)
    : _activation_tensors(std::move(activation_tensors)),
      _allocated_batch_size(0),
      _using_sparsity(true) {}

void ActivationsManager::reallocateForBatch(uint32_t batch_size,
                                            bool use_sparsity) {
  if (batch_size <= _allocated_batch_size && use_sparsity == _using_sparsity) {
    return;
  }

  for (auto& tensor : _activation_tensors) {
    tensor->allocate(batch_size, use_sparsity);
  }
}

const std::vector<tensor::ActivationTensorPtr>&
ActivationsManager::activationTensors() const {
  return _activation_tensors;
}

void ActivationsManager::resetOutputGradients(uint32_t index_in_batch) {
  for (auto& tensor : _activation_tensors) {
    tensor->getVector(index_in_batch).zeroOutGradients();
  }
}

uint32_t ActivationsManager::currentBatchSize() const {
  return _allocated_batch_size;
}

}  // namespace thirdai::bolt::nn::computation_graph