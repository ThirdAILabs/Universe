#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>

namespace thirdai::bolt {

class DotProduct final : public Op,
                         public std::enable_shared_from_this<DotProduct> {
 public:
  static auto make() { return std::shared_ptr<DotProduct>(new DotProduct()); }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                     bool replace_existing_optimizer) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<DotProduct> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr lhs, ComputationPtr rhs);

  static std::string type() { return "dot_product"; }

  void useTorchInitialization() final {}

 private:
  DotProduct() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  static float denseDenseDot(const BoltVector& a, const BoltVector& b);

  static float denseSparseDot(const BoltVector& a, const BoltVector& b);

  static float sparseSparseDot(const BoltVector& a, const BoltVector& b);

  static void denseDenseBackprop(float grad, BoltVector& a, BoltVector& b);

  static void denseSparseBackprop(float grad, BoltVector& a, BoltVector& b);

  static void sparseSparseBackprop(float grad, BoltVector& a, BoltVector& b);
};

using DotProductPtr = std::shared_ptr<DotProduct>;

}  // namespace thirdai::bolt