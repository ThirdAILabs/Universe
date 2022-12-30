#pragma once

#include <bolt/src/nn/loss/ComparativeLoss.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>

namespace thirdai::bolt::nn::loss {

class CategoricalCrossEntropy final : public ComparativeLoss {
 public:
  explicit CategoricalCrossEntropy(tensor::ActivationTensorPtr activations);

  static std::shared_ptr<CategoricalCrossEntropy> make(
      tensor::ActivationTensorPtr activations);

  float gradient(float activation, float label, uint32_t batch_size) final;

 private:
  tensor::ActivationTensorPtr _activations;
};

using CategoricalCrossEntropyPtr = std::shared_ptr<CategoricalCrossEntropy>;

}  // namespace thirdai::bolt::nn::loss