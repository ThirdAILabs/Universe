#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class CosineSimilarity final
    : public Op,
      public std::enable_shared_from_this<CosineSimilarity> {
 public:
  static auto make() {
    return std::shared_ptr<CosineSimilarity>(new CosineSimilarity());
  }

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(autograd::ComputationPtr lhs,
                                 autograd::ComputationPtr rhs);

 private:
  CosineSimilarity() {}

  static float magnitude(const BoltVector& a);

  static float denseDenseSim(const BoltVector& a, const BoltVector& b);

  static float denseSparseSim(const BoltVector& a, const BoltVector& b);

  static float sparseSparseSim(const BoltVector& a, const BoltVector& b);

  static void denseDenseBackprop(float grad, float cos_sim, BoltVector& a,
                                 BoltVector& b);

  static void denseSparseBackprop(float grad, float cos_sim, BoltVector& a,
                                  BoltVector& b);

  static void sparseSparseBackprop(float grad, float cos_sim, BoltVector& a,
                                   BoltVector& b);
};

using CosineSimilarityPtr = std::shared_ptr<CosineSimilarity>;

}  // namespace thirdai::bolt::nn::ops