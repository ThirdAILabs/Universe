#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt::nn::loss {

/**
 * Categorical cross entropy loss function. Same as standard implementation of
 * CCE except it ensures the output activations are >= 1e-6 before taking the
 * log when computing the loss to handle the case in which there is a non-active
 * neuron whose activation is treated as zero.
 */
class MeanSquaredError final : public ComparativeLoss {
 public:
  explicit MeanSquaredError(autograd::ComputationPtr output,
                            autograd::ComputationPtr labels);

  static std::shared_ptr<MeanSquaredError> make(
      autograd::ComputationPtr output, autograd::ComputationPtr labels);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;

  MeanSquaredError() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MeanSquaredErrorPtr = std::shared_ptr<MeanSquaredError>;

}  // namespace thirdai::bolt::nn::loss