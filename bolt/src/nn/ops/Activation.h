#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <archive/src/Archive.h>
#include <cmath>
#include <memory>

namespace thirdai::bolt {

struct ReluImpl {
  static float forward(float x) { return std::max(x, 0.F); }

  static float gradient(float y) { return y > 0 ? 1 : 0; }

  static std::string name() { return "ReLU"; }
};

struct TanhImpl {
  static float forward(float x) { return std::tanh(x); }

  static float gradient(float y) { return (1 - y * y); }

  static std::string name() { return "Tanh"; }
};

template <typename Impl>
class Activation final : public Op,
                         public std::enable_shared_from_this<Activation<Impl>> {
 public:
  static std::shared_ptr<Activation> make();

  void forward(const ComputationList& inputs, TensorPtr& output,
               uint32_t index_in_batch, bool training) final;

  void backpropagate(ComputationList& inputs, TensorPtr& output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(ComputationPtr input);

 private:
  Activation();

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  uint32_t _dim = 0;
};

using Relu = Activation<ReluImpl>;
using ReluPtr = std::shared_ptr<Relu>;

using Tanh = Activation<TanhImpl>;
using TanhPtr = std::shared_ptr<Tanh>;

}  // namespace thirdai::bolt