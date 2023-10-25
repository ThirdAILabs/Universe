#pragma once

#include "Loss.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {

/**
 * Contrastive loss function:
 * L = 0.5 * D(U,V)^2(Y) + (1 - Y) * 0.5 * max(0, m - D(U,V))^2,
 * where D is Euclidean distance and Y is the label (Y = 1 when the points are
 * similar and = 0 when they are dissimilar). See
 * bolt/src/nn/derivations/EuclideanContrastive.md for the gradient derivation.
 */
class EuclideanContrastive final : public Loss {
 public:
  explicit EuclideanContrastive(ComputationPtr output_1,
                                ComputationPtr output_2, ComputationPtr labels,
                                float dissimilar_cutoff_distance);

  static std::shared_ptr<EuclideanContrastive> make(
      const ComputationPtr& output_1, const ComputationPtr& output_2,
      const ComputationPtr& labels, float dissimilar_cutoff_distance);

  void gradients(uint32_t index_in_batch, uint32_t batch_size) const final;

  float loss(uint32_t index_in_batch) const final;

  ComputationList outputsUsed() const final;

  ComputationList labels() const final;

  bool logitsSumToOne() const final { return false; }

 private:
  float euclideanDistanceSquared(uint32_t index_in_batch) const;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  EuclideanContrastive(){};

  ComputationPtr _output_1, _output_2, _labels;
  float _dissimilar_cutoff_distance;
};

using EuclideanContrastivePtr = std::shared_ptr<EuclideanContrastive>;

}  // namespace thirdai::bolt