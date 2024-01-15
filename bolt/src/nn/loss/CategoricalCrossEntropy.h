#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt {

/**
 * Categorical cross entropy loss function. Same as standard implementation of
 * CCE except it ensures the output activations are >= 1e-6 before taking the
 * log when computing the loss to handle the case in which there is a non-active
 * neuron whose activation is treated as zero.
 */
class CategoricalCrossEntropy final : public ComparativeLoss {
 public:
  explicit CategoricalCrossEntropy(ComputationPtr output,
                                   ComputationPtr labels);

  static std::shared_ptr<CategoricalCrossEntropy> make(ComputationPtr output,
                                                       ComputationPtr labels);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;

  CategoricalCrossEntropy() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using CategoricalCrossEntropyPtr = std::shared_ptr<CategoricalCrossEntropy>;

}  // namespace thirdai::bolt