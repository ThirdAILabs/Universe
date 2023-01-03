#pragma once

#include <bolt/src/nn/loss/Loss.h>

namespace thirdai::bolt::nn::loss {

/**
 * Subclass of Loss for loss functions that are performing an elementwise
 * comparison of two vectors of equal dimension. Implementing a loss function of
 * this form just requires inheriting from this class and implmementing a method
 * for returning the gradient given the activation and label for the ith neuron.
 */
class ComparativeLoss : public Loss {
 public:
  explicit ComparativeLoss(tensor::ActivationTensorPtr activations);

  void gradients(uint32_t index_in_batch, uint32_t batch_size) final;

  std::vector<tensor::ActivationTensorPtr> outputsUsed() const final;

  virtual ~ComparativeLoss() = default;

 private:
  /**
   * Helper function to iterate over the activations and labels depending on
   * their sparsities.
   */
  template <bool ACT_DENSE, bool LABEL_DENSE>
  void gradients(BoltVector& activations, const BoltVector& labels,
                 uint32_t batch_size);

  /**
   * This method takes in the activation and label for the ith neuron and should
   * return the loss gradient for that neuron.
   */
  virtual float gradient(float activation, float label,
                         uint32_t batch_size) = 0;

  tensor::ActivationTensorPtr _activations;
};

}  // namespace thirdai::bolt::nn::loss