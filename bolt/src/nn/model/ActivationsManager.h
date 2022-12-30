#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::model {

/**
 * This class manages the ActivationTensors in the model.
 */
class ActivationsManager {
 public:
  explicit ActivationsManager(
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
  uint32_t currentBatchSize() const;

 private:
  std::vector<tensor::ActivationTensorPtr> _activation_tensors;

  uint32_t _allocated_batch_size;
  bool _using_sparsity;
};

}  // namespace thirdai::bolt::nn::model