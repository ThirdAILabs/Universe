#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt {

/**
 * Binary cross entropy loss function. Same as standard implementation of
 * BCE except it clips output activations to [1e-6, 1-1e-6] for stability.
 */
class BinaryCrossEntropy final : public ComparativeLoss {
 public:
  explicit BinaryCrossEntropy(ComputationPtr output, ComputationPtr labels);

  static std::shared_ptr<BinaryCrossEntropy> make(ComputationPtr output,
                                                  ComputationPtr labels);

  proto::bolt::Loss* toProto() const final;

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;

  BinaryCrossEntropy() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using BinaryCrossEntropyPtr = std::shared_ptr<BinaryCrossEntropy>;

}  // namespace thirdai::bolt