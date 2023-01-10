#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::model {

/**
 * This class manages the allocation ActivationTensors in the model. This means
 * that it is responsible for tracking the currenly allocated batch size and
 * whether sparsity is used so that the tensors can be reallocated when either
 * changes. It can also be called to reallocate the tensors when the sparsity of
 * one of the ops changes.
 */
class AllocationManager {
 public:
  explicit AllocationManager(
      std::vector<tensor::ActivationTensorPtr> activation_tensors);

  /**
   * This method will call the allocate(...) method of each tensor if the
   * provided batch size is greater than the currently allocated batch size or
   * if whether or not sparsity is being used is changing.
   */
  void reallocateForBatch(uint32_t batch_size, bool use_sparsity);

  /**
   * Returns all of the ActivationTensors in the model.
   */
  const std::vector<tensor::ActivationTensorPtr>& activationTensors() const;

  /**
   * Sets all of the gradients to 0 for the ith vector of the ActivationTensors.
   * This is called before executing the logic in backpropagate in the model.
   */
  void resetOutputGradients(uint32_t index_in_batch);

  /**
   * Returns the currently allocated batch size.
   */
  constexpr uint32_t currentBatchSize() const { return _current_batch_size; }

  /**
   * Retrieves a tensor by name. Throws if not found.
   */
  tensor::ActivationTensorPtr getTensor(const std::string& name) const;

 private:
  std::vector<tensor::ActivationTensorPtr> _activation_tensors;

  uint32_t _allocated_batch_size;
  uint32_t _current_batch_size;

  bool _using_sparsity;
};

}  // namespace thirdai::bolt::nn::model