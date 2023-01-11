#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt_vector/src/BoltVector.h>

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

  float loss(uint32_t index_in_batch) const final;

  void gradients(uint32_t index_in_batch, uint32_t batch_size) const final;

  std::vector<tensor::ActivationTensorPtr> outputsUsed() const final;

  virtual ~ComparativeLoss() = default;

 private:
  /**
   * Helper functions to iterate over the activations and labels depending on
   * their sparsities.
   */
  template <bool ACT_DENSE, bool LABEL_DENSE>
  float loss(const BoltVector& activations, const BoltVector& labels) const;

  template <bool ACT_DENSE, bool LABEL_DENSE>
  void gradients(BoltVector& activations, const BoltVector& labels,
                 uint32_t batch_size) const;

  /**
   * This method takes in the activation and label for the ith neuron and should
   * return the loss contribution of that neuron.
   */
  virtual float singleLoss(float activation, float label) const = 0;

  /**
   * This method takes in the activation and label for the ith neuron and should
   * return the loss gradient for that neuron.
   */
  virtual float singleGradient(float activation, float label,
                               uint32_t batch_size) const = 0;

  tensor::ActivationTensorPtr _activations;
};

}  // namespace thirdai::bolt::nn::loss