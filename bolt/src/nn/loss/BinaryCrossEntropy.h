#pragma once

#include <bolt/src/nn/loss/ComparativeLoss.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::loss {

/**
 * Binary cross entropy loss function. Same as standard implementation of
 * BCE except it adds clips output activations to [1e-6, 1-1e-6] for stability.
 */
class BinaryCrossEntropy final : public ComparativeLoss {
 public:
  explicit BinaryCrossEntropy(tensor::ActivationTensorPtr activations);

  static std::shared_ptr<BinaryCrossEntropy> make(
      tensor::ActivationTensorPtr activations);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;

  tensor::ActivationTensorPtr _activations;
};

using BinaryCrossEntropyPtr = std::shared_ptr<BinaryCrossEntropy>;

}  // namespace thirdai::bolt::nn::loss