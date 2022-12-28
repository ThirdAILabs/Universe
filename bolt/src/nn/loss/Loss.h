#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::nn::loss {

class Loss {
 public:
  /**
   * Computes the loss for the outputs used by this loss function for the given
   * sample in the batch.
   */
  virtual void computeGradients(uint32_t index_in_batch) = 0;

  /**
   * Returns which outputs in the computation graph have gradients computed by
   * this loss function. This is used to ensure that all of the outputs in the
   * computation graph have gradients computed for them.
   */
  virtual std::vector<tensor::ActivationTensorPtr> outputsUsed() const = 0;

  /**
   * Returns the input tensor for the labels that the loss function is
   * expecting. The labels passed into the computation graph are assigned to the
   * inputs returned by the loss functions in the order that the loss functions
   * are supplied to the computation graph.
   */
  tensor::InputTensorPtr labels() const { return _labels; }

 protected:
  tensor::InputTensorPtr _labels;
};

using LossPtr = std::shared_ptr<Loss>;

}  // namespace thirdai::bolt::nn::loss