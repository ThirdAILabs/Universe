#pragma once

#include <bolt/src/layers/Optimizer.h>
#include <bolt/src/nn/ops/Op.h>

namespace thirdai::bolt::nn::ops {

class LayerNorm final : public Op {
 public:
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

  std::vector<std::vector<float>*> gradients() const final;

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

 private:
  static std::pair<float, float> moments(const BoltVector& vector) {
    float mean = 0.0;
    for (uint32_t i = 0; i < vector.len; i++) {
      mean += vector.activations[i];
    }
    mean /= vector.len;

    float variance = 0.0;
    for (uint32_t i = 0; i < vector.len; i++) {
      float delta = vector.activations[i] - mean;
      variance += (delta * delta);
    }
    variance /= vector.len;

    return {mean, variance};
  }

  float partialDerivativeWRTVariance(const BoltVector& input_vector,
                                     const BoltVector& output_vector,
                                     float mean, float stddev);

  float partialDerivativeWRTMean(const BoltVector& input_vector,
                                 const BoltVector& output_vector, float mean,
                                 float stddev, float partial_wrt_variance);

  std::optional<uint32_t> _dim;

  std::vector<float> _scale;
  std::vector<float> _offset;

  AdamOptimizer _scale_optimizer;
  AdamOptimizer _offset_optimizer;
};

}  // namespace thirdai::bolt::nn::ops