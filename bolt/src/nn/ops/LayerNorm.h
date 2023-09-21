#pragma once

#include <cereal/access.hpp>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

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

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const ComputationList& inputs,
                                   bool use_sparsity) const final;

  void initOptimizer() final;

  void disableSparseParameterUpdates() final;

  void enableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  std::vector<std::vector<float>*> parameters() final;

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  ComputationPtr apply(const ComputationPtr& input);

  const auto& gamma() const { return _gamma; }

  const auto& beta() const { return _beta; }

 private:
  LayerNorm();

  LayerNorm(const float* gamma, const float* beta, size_t dim);

  template <bool DENSE>
  void forward(const BoltVector& input, BoltVector& output);

  template <bool DENSE>
  void backpropagate(BoltVector& input, const BoltVector& output);

  static std::pair<float, float> moments(const BoltVector& vector);

  static constexpr float EPSILON = 1e-6;

  std::vector<float> _gamma;
  std::vector<float> _beta;

  AdamOptimizer _gamma_optimizer;
  AdamOptimizer _beta_optimizer;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using LayerNormPtr = std::shared_ptr<LayerNorm>;

}  // namespace thirdai::bolt