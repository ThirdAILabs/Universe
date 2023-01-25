#pragma once

#include <bolt/src/nn/loss/ComparativeLoss.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::loss {

/**
 * Categorical cross entropy loss function. Same as standard implementation of
 * CCE except it adds 1e-7 to output activations before taking the log when
 * computing the loss to handle the case in which there is a non-active neuron
 * whose activation is treated as zero.
 */
class CategoricalCrossEntropy final : public ComparativeLoss {
 public:
  explicit CategoricalCrossEntropy(autograd::ComputationPtr output,
                                   autograd::ComputationPtr labels);

  static std::shared_ptr<CategoricalCrossEntropy> make(
      autograd::ComputationPtr output, autograd::ComputationPtr labels);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;
};

using CategoricalCrossEntropyPtr = std::shared_ptr<CategoricalCrossEntropy>;

}  // namespace thirdai::bolt::nn::loss