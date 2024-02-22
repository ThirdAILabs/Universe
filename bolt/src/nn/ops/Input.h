#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>

namespace thirdai::bolt {

class Input final : public Op, public std::enable_shared_from_this<Input> {
 public:
  // TODO(Nicholas) add nonzeros as option.
  static ComputationPtr make(uint32_t dim);

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final { return {}; };

  std::vector<std::vector<float>*> parameters() final { return {}; };

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

 private:
  Input(uint32_t dim, std::optional<uint32_t> nonzeros);

  uint32_t _dim;
  std::optional<uint32_t> _nonzeros;

  Input() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using InputPtr = std::shared_ptr<Input>;

}  // namespace thirdai::bolt