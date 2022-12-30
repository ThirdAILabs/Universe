#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/ops/Op.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

class FullyConnected final : public Op {
 public:
  static tensor::ActivationTensorPtr apply(
      std::shared_ptr<FullyConnectedLayer> kernel, tensor::Tensor* input,
      std::string name);

  void forward(uint32_t index_in_batch) final;

  void backpropagate(uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void disableSparseParameterUpdates() final;

  std::vector<tensor::Tensor*> inputs() const final;

  std::vector<tensor::ActivationTensorPtr> outputs() const final;

  void notifyInputSparsityChange() final {}

  void summary(std::ostream& summary) const final;

  /**
   * Returns the dimensions of the layer as {dim, input_dim}.
   */
  std::vector<uint32_t> dimensions() const;

  /**
   * Returns a non-owning pointer to the weights.
   */
  const float* weightsPtr() const;

  /**
   * Returns a non-owning pointer to the biases.
   */
  const float* biasesPtr() const;

 private:
  FullyConnected(std::shared_ptr<FullyConnectedLayer> kernel,
                 tensor::Tensor* input, std::string name);

  std::shared_ptr<FullyConnectedLayer> _kernel;

  tensor::Tensor* _input;

  tensor::ActivationTensorPtr _output;
};

using FullyConnectedPtr = std::shared_ptr<FullyConnected>;

class FullyConnectedFactory {
 public:
  FullyConnectedFactory(uint32_t dim, float sparsity, std::string activation,
                        SamplingConfigPtr sampling);

  tensor::ActivationTensorPtr apply(tensor::TensorPtr& input);

 private:
  uint32_t _dim;
  float _sparsity;
  std::string _activation;
  SamplingConfigPtr _sampling;

  std::shared_ptr<FullyConnectedLayer> _kernel;

  std::string _name;
};

}  // namespace thirdai::bolt::nn::ops