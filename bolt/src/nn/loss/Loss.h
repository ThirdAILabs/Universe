#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt::nn::loss {

class Loss {
 public:
  /**
   * Computes the loss gradients for the outputs used by this loss function for
   * the given sample in the batch and sets the gradients of the output
   * tensor(s) used in this loss function.
   */
  virtual void gradients(uint32_t index_in_batch,
                         uint32_t batch_size) const = 0;

  /**
   * Computes the loss for the given sample in the batch.
   */
  virtual float loss(uint32_t index_in_batch) const = 0;

  /**
   * Returns which outputs in the model have gradients computed by this loss
   * function. This is used to ensure that all of the outputs in the model have
   * gradients computed for them.
   */
  virtual autograd::ComputationList outputsUsed() const = 0;

  /**
   * Returns the labels that the loss function is expecting. These are any
   * inputs to the loss function that do not come from the model itself. For
   * instance a classical label vector, or even per sample weights for the loss.
   */
  virtual autograd::ComputationList labels() const = 0;

  virtual ~Loss() = default;
};

using LossPtr = std::shared_ptr<Loss>;

}  // namespace thirdai::bolt::nn::loss