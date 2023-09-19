#pragma once

#include "Loss.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt {

class CosineContrastive final : public Loss {
 public:
  explicit CosineContrastive(ComputationPtr output_1, ComputationPtr output_2,
                             ComputationPtr labels,
                             float dissimilar_cutoff_distance);

  static std::shared_ptr<CosineContrastive> make(
      const ComputationPtr& output_1, const ComputationPtr& output_2,
      const ComputationPtr& labels, float dissimilar_cutoff_distance);

  void gradients(uint32_t index_in_batch, uint32_t batch_size) const final;

  float loss(uint32_t index_in_batch) const final;

  ComputationList outputsUsed() const final;

  ComputationList labels() const final;

 private:
  float cosineSim(uint32_t index_in_batch) const;

  static float magnitude(const BoltVector& a);

  static float denseDenseSim(const BoltVector& a, const BoltVector& b);

  static float denseSparseSim(const BoltVector& a, const BoltVector& b);

  static float sparseSparseSim(const BoltVector& a, const BoltVector& b);

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  CosineContrastive(){};

  ComputationPtr _output_1, _output_2, _labels;
  float _dissimilar_cutoff_distance;
};

using CosineContrastivePtr = std::shared_ptr<CosineContrastive>;

}  // namespace thirdai::bolt