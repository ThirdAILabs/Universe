#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>

namespace thirdai::bolt::nn::ops {

class FullyConnected final
    : public Op,
      public std::enable_shared_from_this<FullyConnected> {
 public:
  static std::shared_ptr<FullyConnected> make(
      uint32_t dim, uint32_t input_dim, float sparsity,
      const std::string& activation, SamplingConfigPtr sampling,
      uint32_t rebuild_hash_tables = std::numeric_limits<uint32_t>::max(),
      uint32_t reconstruct_hash_functions =
          std::numeric_limits<uint32_t>::max());

  /**
   * Inputs will always have size=1, except if the op yields an output, in which
   * case the labels will be passed in as a second input so that the layer can
   * ensure that the label neurons are among the active neurons set if it's
   * sparse.
   */
  void forward(const tensor::TensorList& inputs,
               tensor::ActivationTensor* output, uint32_t index_in_batch,
               bool training) final;

  void backpropagate(tensor::TensorList& inputs,
                     tensor::ActivationTensor* output,
                     uint32_t index_in_batch) final;

  void updateParameters(float learning_rate, uint32_t train_steps) final;

  uint32_t numNonzerosInOutput(const tensor::TensorList& inputs,
                               bool use_sparsity) const final;

  void disableSparseParameterUpdates() final;

  void summary(std::ostream& summary, const tensor::TensorList& inputs,
               const tensor::ActivationTensor* output) const final;

  /**
   * Applies the op to an input tensor and yields a new output tensor.
   */
  tensor::ActivationTensorPtr apply(tensor::TensorPtr input);

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
  FullyConnected(
      uint32_t dim, uint32_t input_dim, float sparsity,
      const std::string& activation, SamplingConfigPtr sampling,
      uint32_t rebuild_hash_tables = std::numeric_limits<uint32_t>::max(),
      uint32_t reconstruct_hash_functions =
          std::numeric_limits<uint32_t>::max());

  std::shared_ptr<FullyConnectedLayer> _kernel;

  uint32_t _rebuild_hash_tables;
  uint32_t _reconstruct_hash_functions;
  uint32_t _updates_since_rebuild_hash_tables;
  uint32_t _updates_since_reconstruct_hash_functions;
};

using FullyConnectedPtr = std::shared_ptr<FullyConnected>;

}  // namespace thirdai::bolt::nn::ops