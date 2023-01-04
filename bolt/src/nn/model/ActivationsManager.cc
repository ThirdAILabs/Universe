#include "ActivationsManager.h"

namespace thirdai::bolt::nn::model {

ActivationsManager::ActivationsManager(
    std::vector<tensor::ActivationTensorPtr> activation_tensors)
    : _activation_tensors(std::move(activation_tensors)),
      _allocated_batch_size(0),
      _current_batch_size(0),
      _using_sparsity(true) {}

void ActivationsManager::reallocateForBatch(uint32_t batch_size,
                                            bool use_sparsity) {
  _current_batch_size = batch_size;
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

tensor::ActivationTensorPtr ActivationsManager::getTensor(
    const std::string& name) const {
  for (const auto& tensor : _activation_tensors) {
    if (tensor->name() == name) {
      return tensor;
    }
  }
  throw std::invalid_argument("Could not find tensor with name '" + name +
                              "'.");
}

}  // namespace thirdai::bolt::nn::model