#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::loss {

class Loss {
 public:
  /**
   * Computes the loss gradients for the outputs used by this loss function for
   * the given sample in the batch.
   */
  virtual void gradients(uint32_t index_in_batch, uint32_t batch_size) const = 0;

  /**
   * Computes the loss for the given sample in the batch.
   */
  virtual float loss(uint32_t index_in_batch) const = 0;

  /**
   * Returns which outputs in the model have gradients computed by this loss
   * function. This is used to ensure that all of the outputs in the model have
   * gradients computed for them.
   */
  virtual std::vector<tensor::ActivationTensorPtr> outputsUsed() const = 0;

  /**
   * Returns the input tensor for the labels that the loss function is
   * expecting. The labels passed into the model are assigned to the inputs
   * returned by the loss functions in the order that the loss functions are
   * supplied to the model.
   */
  tensor::InputTensorPtr labels() const { return _labels; }

  virtual ~Loss() = default;

 protected:
  tensor::InputTensorPtr _labels;
};

using LossPtr = std::shared_ptr<Loss>;

}  // namespace thirdai::bolt::nn::loss