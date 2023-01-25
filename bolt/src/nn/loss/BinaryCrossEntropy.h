#pragma once

#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt::nn::loss {

/**
 * Binary cross entropy loss function. Same as standard implementation of
 * BCE except it adds clips output activations to [1e-6, 1-1e-6] for stability.
 */
class BinaryCrossEntropy final : public ComparativeLoss {
 public:
  explicit BinaryCrossEntropy(autograd::ComputationPtr output,
                              autograd::ComputationPtr labels);

  static std::shared_ptr<BinaryCrossEntropy> make(
      autograd::ComputationPtr output, autograd::ComputationPtr labels);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;
};

using BinaryCrossEntropyPtr = std::shared_ptr<BinaryCrossEntropy>;

}  // namespace thirdai::bolt::nn::loss