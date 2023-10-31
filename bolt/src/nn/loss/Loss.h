#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt {

/**
 * Loss functions are used to compute the gradients of the terminal computations
 * in the computation graph (computations with no successors). The Losses should
 * be constructed with a combination of terminal computations in the computation
 * graph and labels. There is no restriction on how many computations and labels
 * the Loss can use.
 */
class Loss {
 public:
  /**
   * Computes the loss gradients for the outputs used by this loss function for
   * the given sample in the batch and sets the gradients of the output
   * tensor(s) used in this loss function. The training loop updates in the
   * direction of this gradient, so if you are trying to minimize the loss make
   * sure to use the negative partial derivative. You should also make sure to
   * divide the final value by the batch size.
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
  virtual ComputationList outputsUsed() const = 0;

  /**
   * Returns the labels that the loss function is expecting. These are any
   * inputs to the loss function that do not come from the model itself. For
   * instance a classical label vector, or even per sample weights for the loss.
   */
  virtual ComputationList labels() const = 0;

  virtual ~Loss() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using LossPtr = std::shared_ptr<Loss>;

}  // namespace thirdai::bolt