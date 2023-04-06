#pragma once

#include "Loss.h"
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::loss {

/**
 * Contrastive loss function, 0.5D^2(1 - Y) + Ymax(0, m - D)^2. See LeCun:
 * http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.
 */
class Contrastive final : public Loss {
 public:
  explicit Contrastive(autograd::ComputationPtr output_1,
                       autograd::ComputationPtr output_2,
                       autograd::ComputationPtr labels,
                       float dissimilar_cutoff_margin);

  static std::shared_ptr<Contrastive> make(autograd::ComputationPtr output_1,
                                           autograd::ComputationPtr output_2,
                                           autograd::ComputationPtr labels,
                                           float dissimilar_cutoff_margin);

  void gradients(uint32_t index_in_batch, uint32_t batch_size) const final;

  float loss(uint32_t index_in_batch) const final;

  autograd::ComputationList outputsUsed() const final;

  autograd::ComputationList labels() const final;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

 private:
  float euclideanDistanceSquared(uint32_t index_in_batch) const;

  autograd::ComputationPtr _output_1, _output_2, _labels;
  float _dissimilar_cutoff_margin;
};

using ContrastivePtr = std::shared_ptr<Contrastive>;

}  // namespace thirdai::bolt::nn::loss