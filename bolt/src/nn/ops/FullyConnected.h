#pragma once

#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>

namespace thirdai::bolt::nn::ops {

class FullyConnected final
    : public Op,
      public std::enable_shared_from_this<FullyConnected> {
 public:
  // TODO(Nicholas): rebuild_hash_tables & reconstruct_hash_functions should be
  // moved to the sampling config once bolt v1 is depreciated and there are no
  // compatability concerns.
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

  void summary(std::ostream& summary, const autograd::ComputationList& inputs,
               const autograd::Computation* output) const final;

  /**
   * Applies the op to an input tensor and yields a new output tensor. Used to
   * add the op to a computation graph.
   */
  autograd::ComputationPtr apply(autograd::ComputationPtr input);

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

  FullyConnected() {}

  friend class cereal::access;

  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

using FullyConnectedPtr = std::shared_ptr<FullyConnected>;

}  // namespace thirdai::bolt::nn::ops