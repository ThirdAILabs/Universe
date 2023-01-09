#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <limits>
#include <memory>

namespace thirdai::bolt::nn::ops {

class FullyConnected final : public Op {
 public:
  static tensor::ActivationTensorPtr apply(
      std::shared_ptr<FullyConnectedLayer> kernel, tensor::Tensor* input,
      std::string name, uint32_t rebuild_hash_tables,
      uint32_t reconstruct_hash_functions);

  void forward(uint32_t index_in_batch, bool training) final;

  void backpropagate(uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  void disableSparseParameterUpdates() final;

  std::vector<tensor::Tensor*> inputs() const final;

  std::vector<tensor::ActivationTensorPtr> outputs() const final;

  void notifyInputSparsityChange() final {}

  void summary(std::ostream& summary) const final;

  /**
   * This is so that during training if a fully connected layer is sparse and
   * yields an output that the labels can be selected as part of the active
   * neurons. The method sets the InputTensor that will be used to access the
   * labels. This should only be called if the FullyConnected op yields an
   * output.
   */
  void setLabels(tensor::InputTensorPtr labels);

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
                 tensor::Tensor* input, std::string name,
                 uint32_t rebuild_hash_tables,
                 uint32_t reconstruct_hash_functions);

  std::shared_ptr<FullyConnectedLayer> _kernel;
  uint32_t _rebuild_hash_tables;
  uint32_t _reconstruct_hash_functions;
  uint32_t _updates_since_rebuild_hash_tables;
  uint32_t _updates_since_reconstruct_hash_functions;

  tensor::Tensor* _input;

  tensor::ActivationTensorPtr _output;

  // See comment for setLabels for what this is used for.
  tensor::InputTensorPtr _labels;
};

using FullyConnectedPtr = std::shared_ptr<FullyConnected>;

class FullyConnectedFactory {
 public:
  FullyConnectedFactory(
      uint32_t dim, float sparsity, std::string activation,
      SamplingConfigPtr sampling,
      uint32_t rebuild_hash_tables = std::numeric_limits<uint32_t>::max(),
      uint32_t reconstruct_hash_functions =
          std::numeric_limits<uint32_t>::max());

  tensor::ActivationTensorPtr apply(const tensor::TensorPtr& input);

 private:
  uint32_t _dim;
  float _sparsity;
  std::string _activation;
  SamplingConfigPtr _sampling;
  uint32_t _rebuild_hash_tables;
  uint32_t _reconstruct_hash_functions;

  std::shared_ptr<FullyConnectedLayer> _kernel;

  std::string _name;
};

}  // namespace thirdai::bolt::nn::ops