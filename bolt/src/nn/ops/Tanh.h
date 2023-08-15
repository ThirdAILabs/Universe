#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt {

class Tanh final : public Op, public std::enable_shared_from_this<Tanh> {
 public:
  static std::shared_ptr<Tanh> make();

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr input);

 private:
  Tanh();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  uint32_t _dim = 0;
};

using TanhPtr = std::shared_ptr<Tanh>;

}  // namespace thirdai::bolt