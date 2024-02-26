#pragma once

#include <cereal/access.hpp>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <archive/src/Archive.h>
#include <limits>
#include <memory>

namespace thirdai::bolt {

class FullyConnected final
    : public Op,
      public std::enable_shared_from_this<FullyConnected> {
 public:
  // TODO(Nicholas): rebuild_hash_tables & reconstruct_hash_functions should be
  // moved to the sampling config once bolt v1 is depreciated and there are no
  // compatability concerns.
  static std::shared_ptr<FullyConnected> make(
      uint32_t dim, uint32_t input_dim, float sparsity,
      const std::string& activation, SamplingConfigPtr sampling = nullptr,
      bool use_bias = true, uint32_t rebuild_hash_tables = 4,
      uint32_t reconstruct_hash_functions = 100);

  /**
   * Inputs will always have size=1, except if the op yields an output, in which
   * case the labels will be passed in as a second input so that the layer can
   * ensure that the label neurons are among the active neurons set if it's
   * sparse.
   */
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

  static std::shared_ptr<FullyConnected> fromArchive(
      const ar::Archive& archive);

  void summary(std::ostream& summary, const ComputationList& inputs,
               const Computation* output) const final;

  void setSerializeOptimizer(bool should_serialize_optimizer) final;

  void registerModel(const std::weak_ptr<Model>& new_model) final;

  std::vector<std::pair<std::string, double>> parameterAndGradNorms()
      const final;

  /**
   * Applies the op to an input tensor and yields a new output tensor. Used to
   * add the op to a computation graph.
   */
  ComputationPtr apply(ComputationPtr input);

  /**
   * Returns the input dim of the fully connected layer.
   */
  uint32_t inputDim() const;

  /**
   * Returns a non-owning pointer to the weights.
   */
  const float* weightsPtr() const;

  /**
   * Returns a non-owning pointer to the biases.
   */
  const float* biasesPtr() const;

  std::shared_ptr<FullyConnectedLayer> kernel() const;

  /**
   * Freezes all hash tables in the model. The parameter
   * insert_labels_if_not_found controls if label neurons should be inserted
   * into the hash tables at the buckets that were probed when they are not
   * found during training.
   */
  void freezeHashTables(bool insert_labels_if_not_found);

  /**
   * Unfreezes all hash tables in the model.
   */
  void unfreezeHashTables();

  void setWeights(const float* new_weights);

  void setBiases(const float* new_biases);

  void reBuildHashFunction();

  std::pair<hashing::HashFunctionPtr, hashtable::SampledHashTablePtr>
  getHashTable() const;

  void setHashTable(hashing::HashFunctionPtr hash_fn,
                    hashtable::SampledHashTablePtr hash_table);

  /**
   * Autotunes how often the hash tables and hash functions are rebuilt using
   * the number of batches in the dataset and the batch size.
   */
  void autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size);

  static auto cast(const OpPtr& op) {
    return std::dynamic_pointer_cast<FullyConnected>(op);
  }

  float getSparsity() { return _kernel->getSparsity(); }

  void setSparsity(float sparsity, bool rebuild_hash_tables,
                   bool experimental_autotune);

  uint32_t getRebuildHashTables() const { return _rebuild_hash_tables; }

  uint32_t getReconstructHashFunctions() const {
    return _reconstruct_hash_functions;
  }

  static std::string type() { return "fc"; }

 private:
  FullyConnected(
      uint32_t dim, uint32_t input_dim, float sparsity,
      const std::string& activation, SamplingConfigPtr sampling = nullptr,
      bool use_bias = true,
      uint32_t rebuild_hash_tables = std::numeric_limits<uint32_t>::max(),
      uint32_t reconstruct_hash_functions =
          std::numeric_limits<uint32_t>::max());

  explicit FullyConnected(const ar::Archive& archive);

  std::shared_ptr<FullyConnectedLayer> _kernel;

  uint32_t _rebuild_hash_tables;
  uint32_t _reconstruct_hash_functions;
  uint32_t _updates_since_rebuild_hash_tables;
  uint32_t _updates_since_reconstruct_hash_functions;

  // This does not need to be serialized because models will register with their
  // ops again once loaded.
  std::vector<std::weak_ptr<Model>> _models_using_op;

  FullyConnected() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using FullyConnectedPtr = std::shared_ptr<FullyConnected>;

}  // namespace thirdai::bolt
