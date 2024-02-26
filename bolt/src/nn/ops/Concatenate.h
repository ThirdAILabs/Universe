#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <archive/src/Archive.h>
#include <memory>

namespace thirdai::bolt {

class Concatenate final : public Op,
                          public std::enable_shared_from_this<Concatenate> {
 public:
  static std::shared_ptr<Concatenate> make();

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final {
    (void)learning_rate;
    (void)train_steps;
  }

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory,
                     bool replace_existing_optimizer) final {
    (void)optimizer_factory;
    (void)replace_existing_optimizer;
  }

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; };

  std::vector<std::vector<float>*> parameters() final { return {}; };

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<Concatenate> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(const ComputationList& inputs);

  static std::string type() { return "concat"; }

 private:
  Concatenate();

  std::vector<uint32_t> _input_dims;
  std::vector<uint32_t> _neuron_offsets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using ConcatenatePtr = std::shared_ptr<Concatenate>;

}  // namespace thirdai::bolt