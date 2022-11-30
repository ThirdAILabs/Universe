#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::bolt {

class ConcatenateNode final
    : public Node,
      public std::enable_shared_from_this<ConcatenateNode> {
 private:
  ConcatenateNode() : _compiled(false){};

 public:
  static std::shared_ptr<ConcatenateNode> make() {
    return std::shared_ptr<ConcatenateNode>(new ConcatenateNode());
  }

  std::shared_ptr<ConcatenateNode> setConcatenatedNodes(
      const std::vector<NodePtr>& nodes);

  uint32_t outputDim() const final;

  bool isInputNode() const final { return false; }

  void initOptimizer() final {}

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final;

  uint32_t numNonzerosInOutputImpl() const final {
    return _batch_processing_state->num_nonzeros_in_concatenation;
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final;

  void backpropagateImpl(uint32_t vec_index) final;

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
    // NOOP because a concatenation layer has no parameters
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return _batch_processing_state->outputs[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final {
    _batch_processing_state = std::nullopt;
  }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return _graph_state->inputs;
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final;

  std::string type() const final { return "concat"; }

  static void verifyNoInputNodes(const std::vector<NodePtr>& nodes);

  static bool concatenationHasSparseNode(const std::vector<NodePtr>& nodes) {
    return std::any_of(nodes.begin(), nodes.end(), nodeIsSparse);
  }

  static bool nodeIsSparse(const NodePtr& node) {
    return node->numNonzerosInOutput() < node->outputDim();
  }

  static std::vector<uint32_t> getPositionalOffsets(
      const std::vector<NodePtr>& nodes, bool use_sparsity);

  static BoltBatch generateBatch(
      bool use_sparsity, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& neuron_index_offsets, uint32_t batch_size);

  // This method prefills the active_neurons for the concatenated output,
  // assuming that each vector is dense. This has no impact on sparse parts
  // of the concatenation since they will override the preset active_neurons.
  static void fillSparseBatchWithConsecutiveIndices(
      BoltBatch& batch, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& neuron_index_offsets);

  NodeState getState() const final;

  struct GraphState {
    // Constructor for cereal.
    GraphState() {}

    // We have this constructor so clang tidy can check variable names
    GraphState(std::vector<NodePtr> inputs,
               std::vector<uint32_t> neuron_index_offsets,
               uint32_t concatenated_dense_dim)
        : inputs(std::move(inputs)),
          neuron_index_offsets(std::move(neuron_index_offsets)),
          concatenated_dense_dim(concatenated_dense_dim){};

    // The input Nodes we are concatenating
    std::vector<NodePtr> inputs;
    /*
     * The ith element in neuron_index_offsets is the "index offset" for the ith
     * input vector in the output concatenated vector. In other words,
     * neuron_index_offsets[i] is how much we need to add to the indices of
     * (index, activation) neuron pairs of the ith input vector to convert them
     * to their correct values in the concatenated output vector. Contains
     * len(inputs) + 1 number of elements, as we use the pattern where the last
     * item would be the neuron index offset for the "next" input if there was
     * one (this allows you to find the dense dim of the ith input vector by
     * doing neuron_index_offsets[i + 1] - neuron_index_offsets[i]).
     */
    std::vector<uint32_t> neuron_index_offsets;
    // Also equals the last element of neuron_index_offsets. This is the dense
    // dimension of the output vector
    uint32_t concatenated_dense_dim;

   private:
    friend class cereal::access;
    template <class Archive>
    void serialize(Archive& archive) {
      archive(inputs, neuron_index_offsets, concatenated_dense_dim);
    }
  };

  struct BatchProcessingState {
    // We have this constructor so clang tidy can check variable names
    BatchProcessingState(std::vector<uint32_t> positional_offsets,
                         BoltBatch outputs,
                         uint32_t num_nonzeros_in_concatenation)
        : positional_offsets(std::move(positional_offsets)),
          outputs(std::move(outputs)),
          num_nonzeros_in_concatenation(num_nonzeros_in_concatenation){};

    /*
     * The ith element in positional_offets is the "positional offset" for the
     * ith input vector in the output concatenated vector. In other words,
     * activations[j] for input vector i will be at
     * activations[j + positional_offsets[i]] in the output concatenated vector,
     * and similar for the gradient and active neurons (the corresponding active
     * neurons will be offset by neuron_index_offsets[i], see the comment for
     * neuron_index_offsets). Uses the same pattern as neuron_index_offsets
     * where this contains len(inputs) + 1 number of elements.
     */
    std::vector<uint32_t> positional_offsets;
    // Each vector in the batch has dimension num_nonzeros_in_concatenation
    BoltBatch outputs;
    // Also equals the last element of positional_offsets. This is the number of
    // non zeros in the output vector (i.e. the value of outputs[i].len) for
    // all i).
    uint32_t num_nonzeros_in_concatenation;
  };

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _graph_state, _compiled);
  }

  std::optional<GraphState> _graph_state;
  bool _compiled = false;
  std::optional<BatchProcessingState> _batch_processing_state;
};

}  // namespace thirdai::bolt
