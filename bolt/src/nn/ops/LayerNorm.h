#pragma once

#include <cereal/access.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <memory>
#include <vector>

namespace thirdai::bolt {

class LayerNorm final : public Op,
                        public std::enable_shared_from_this<LayerNorm> {
 public:
  static std::shared_ptr<LayerNorm> make();

  static std::shared_ptr<LayerNorm> make(const float* gamma, const float* beta,
                                         size_t dim);

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

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  ComputationPtr applyToInputs(const ComputationList& inputs) final;

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::shared_ptr<LayerNorm> fromArchive(const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(const ComputationPtr& input);

  const auto& gamma() const { return _gamma; }

  const auto& beta() const { return _beta; }

  static std::string type() { return "layer_norm"; }

 private:
  LayerNorm();

  explicit LayerNorm(const ar::Archive& archive);

  LayerNorm(const float* gamma, const float* beta, size_t dim);

  template <bool DENSE>
  void forward(const BoltVector& input, BoltVector& output);

  template <bool DENSE>
  void backpropagate(BoltVector& input, const BoltVector& output);

  static std::pair<float, float> moments(const BoltVector& vector);

  static constexpr float EPSILON = 1e-6;

  std::vector<float> _gamma;
  std::vector<float> _beta;

  std::vector<float> _gamma_gradients;
  std::vector<float> _beta_gradients;

  OptimizerPtr _gamma_optimizer;
  OptimizerPtr _beta_optimizer;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using LayerNormPtr = std::shared_ptr<LayerNorm>;

}  // namespace thirdai::bolt