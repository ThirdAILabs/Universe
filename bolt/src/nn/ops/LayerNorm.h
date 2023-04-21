#pragma once

#include <cereal/access.hpp>
#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>
#include <utils/CerealWrapper.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class LayerNorm final : public Op,
                        public std::enable_shared_from_this<LayerNorm> {
 public:
  static std::shared_ptr<LayerNorm> make();

  void forward(const autograd::ComputationList& inputs,
               tensor::TensorPtr& output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(autograd::ComputationList& inputs,
                     tensor::TensorPtr& output, uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t dim() const final;

  std::optional<uint32_t> nonzeros(const autograd::ComputationList& inputs,
                                   bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  std::vector<std::vector<float>*> gradients() final;

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  autograd::ComputationPtr apply(const autograd::ComputationPtr& input);

 private:
  LayerNorm();

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

}  // namespace thirdai::bolt::nn::ops

CEREAL_REGISTER_TYPE_HEADER(thirdai::bolt::nn::ops::LayerNorm)