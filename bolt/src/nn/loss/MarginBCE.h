#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/loss/ComparativeLoss.h>

namespace thirdai::bolt {

class MarginBCE final : public ComparativeLoss {
 public:
  explicit MarginBCE(ComputationPtr output, ComputationPtr labels, float margin = 0.1F);

  static std::shared_ptr<MarginBCE> make(ComputationPtr output,
                                         ComputationPtr labels, float margin = 0.1F);

 private:
  float singleGradient(float activation, float label,
                       uint32_t batch_size) const final;

  float singleLoss(float activation, float label) const final;

  float _positive_margin = 0.1;
  float _negative_margin = 0.1;
  bool _bound = true;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  MarginBCE() {}
};

using MarginBCEPtr = std::shared_ptr<MarginBCE>;

}  // namespace thirdai::bolt