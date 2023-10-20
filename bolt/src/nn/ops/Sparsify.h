#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <cereal/access.hpp>
#include <memory>

namespace thirdai::bolt {

class Sparsify final : public Op,
                       public std::enable_shared_from_this<Sparsify> {
 private:
  explicit Sparsify(float sparsity) : _sparsity(sparsity) {}

 public:
  static auto make(float sparsity) {
    return std::shared_ptr<Sparsify>(new Sparsify(sparsity));
  }

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final {}

  void enableSparseParameterUpdates() final {}

  std::vector<std::vector<float>*> gradients() final { return {}; }

  std::vector<std::vector<float>*> parameters() final { return {}; }

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr input);

 private:
  size_t _dim = 0;
  float _sparsity;

  Sparsify() {};

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

};

using SparsifyPtr = std::shared_ptr<Sparsify>;

}  // namespace thirdai::bolt