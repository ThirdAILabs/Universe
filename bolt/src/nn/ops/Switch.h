#pragma once

#include <cereal/access.hpp>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <limits>
#include <memory>
#include <vector>

namespace thirdai::bolt {

/**
 * An wrapper around n_layer FullyConnected ops that allows the model to
 * choose which FullyConnected op to use based on the input.
 */
class Switch final : public FCKernelOp,
                     public std::enable_shared_from_this<Switch> {
 public:
  // TODO(Nicholas / Geordie): rebuild_hash_tables & reconstruct_hash_functions
  // should be moved to the sampling config once bolt v1 is depreciated and
  // there are no compatability concerns.
  static std::shared_ptr<Switch> make(
      uint32_t n_layers, uint32_t dim, uint32_t input_dim, float sparsity,
      const std::string& activation,
      const SamplingConfigPtr& sampling = nullptr, bool use_bias = true,
      uint32_t rebuild_hash_tables = 4,
      uint32_t reconstruct_hash_functions = 100);

  /**
   * `inputs` will either have size=2 or size=3.
   * The first input is expected to yield a single-element sparse vector where
   * the element's index determines which layer to use.
   * The second input is passed through the selected layer; this is the one that
   * is actually used for computations.
   * The third input contains labels and is only present when this op is one of
   * the model's outputs. This allows the layer to include label neurons in the
   * active neurons set when the layer is sparse.
   */
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

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  /**
   * Applies the op to an input tensor and yields a new output tensor. Used to
   * add the op to a computation graph.
   */
  ComputationPtr apply(ComputationPtr index, ComputationPtr input);

  /**
   * Returns the input dim of the switch connected layer.
   */
  uint32_t inputDim() const;

  /**
   * Freezes all hash tables in the model. The parameter
   * insert_labels_if_not_found controls if label neurons should be inserted
   * into the hash tables at the buckets that were probed when they are not
   * found during training.
   */
  void freezeHashTables(bool insert_labels_if_not_found);

  void setWeights(uint32_t layer_id, const float* weights_to_set);

  void setBiases(uint32_t layer_id, const float* biases_to_set);

  void setSparsity(float sparsity, bool rebuild_hash_tables,
                   bool experimental_autotune) final;

  void unfreezeHashTables() final;

  /**
   * Autotunes how often the hash tables and hash functions are rebuilt using
   * the number of batches in the dataset and the batch size.
   */
  void autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size) final;

  static auto cast(const OpPtr& op) {
    return std::dynamic_pointer_cast<Switch>(op);
  }

  ActivationFunction getActivationFunction() const final {
    return _fc_ops.front()->getActivationFunction();
  }

 private:
  Switch(uint32_t n_layers, uint32_t dim, uint32_t input_dim, float sparsity,
         const std::string& activation, const SamplingConfigPtr& sampling,
         bool use_bias = true,
         uint32_t rebuild_hash_tables = std::numeric_limits<uint32_t>::max(),
         uint32_t reconstruct_hash_functions =
             std::numeric_limits<uint32_t>::max());

  FullyConnectedPtr getFcOpForInputs(const ComputationList& inputs,
                                     uint32_t index_in_batch);

  FullyConnectedPtr getFcOpById(uint32_t layer_id);

  static ComputationList fcInputs(const ComputationList& inputs);

  std::vector<FullyConnectedPtr> _fc_ops;

  Switch() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using SwitchPtr = std::shared_ptr<Switch>;

}  // namespace thirdai::bolt