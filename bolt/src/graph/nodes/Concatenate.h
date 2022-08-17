#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
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
 public:
  ConcatenateNode() : _compiled(false){};

  std::shared_ptr<ConcatenateNode> setConcatenatedNodes(
      const std::vector<NodePtr>& nodes) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "Have already set the incoming concatenated nodes for this "
          "concatenation layer");
    }
    if (nodes.size() < 2) {
      throw exceptions::GraphCompilationFailure(
          "Must concatenate at least two nodes, found " +
          std::to_string(nodes.size()));
    }

    verifyNoInputNodes(nodes);

    std::vector<uint32_t> neuron_index_offsets = {0};
    uint32_t output_dim = 0;
    for (const auto& node : nodes) {
      output_dim += node->outputDim();
      neuron_index_offsets.push_back(output_dim);
    }

    _graph_state = GraphState(/* inputs = */ nodes,
                              /* neuron_index_offsets = */ neuron_index_offsets,
                              /* concatenated_dense_dim = */ output_dim);

    return shared_from_this();
  }

  uint32_t outputDim() const final {
    if (getState() == NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "Cannot get the output dim for this concatenation layer because the "
          "incoming concatenated nodes have not been set yet");
    }
    return _graph_state->concatenated_dense_dim;
  }

  bool isInputNode() const final { return false; }

  // initializes any state needed for training (like the optimizer)
  void prepareForTraining() final {}

 private:
  void compileImpl() final { _compiled = true; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    const auto& concatenated_nodes = _graph_state->inputs;

    bool sparse_concatenation = concatenationHasSparseNode(concatenated_nodes);
    if (sparse_concatenation && !use_sparsity) {
      throw exceptions::NodeStateMachineError(
          "Input to concatenation contains a sparse vector but use_sparsity in "
          "this call to prepareForBatchProcessing is false.");
    }

    std::vector<uint32_t> positional_offsets = getPositionalOffsets(
        concatenated_nodes, /* use_sparsity = */ sparse_concatenation);
    BoltBatch new_concatenated_batch = generateBatch(
        /* use_sparsity = */ sparse_concatenation,
        /* positional_offsets = */ positional_offsets,
        /* neuron_index_offsets = */ _graph_state->neuron_index_offsets,
        /* batch_size = */ batch_size);

    uint32_t num_nonzeros_in_concatenation = positional_offsets.back();
    _batch_processing_state = BatchProcessingState(
        /* positional_offsets = */ std::move(positional_offsets),
        /* outputs = */ std::move(new_concatenated_batch),
        /* num_nonzeros_in_concatenation = */ num_nonzeros_in_concatenation);
  }

  uint32_t numNonzerosInOutputImpl() const final {
    return _batch_processing_state->num_nonzeros_in_concatenation;
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    // We currently do not allow a concatenation layer to be the last
    // layer in the graph.
    // TODO(josh/nick): Add support for n sets of outputs, and if users want
    // a concatenation layer as the last layer they can split the labels
    assert(labels == nullptr);
    (void)labels;

    const BoltVector& output_vector = getOutputVectorImpl(vec_index);
    std::fill_n(output_vector.gradients, output_vector.len, 0);

    const auto& concatenated_nodes = _graph_state->inputs;
    const auto& positional_offsets =
        _batch_processing_state->positional_offsets;
    const auto& neuron_index_offsets = _graph_state->neuron_index_offsets;

    for (uint32_t input_node_id = 0; input_node_id < concatenated_nodes.size();
         input_node_id++) {
      const auto& current_input_node = concatenated_nodes.at(input_node_id);
      BoltVector& current_input =
          current_input_node->getOutputVector(vec_index);
      uint32_t start_position = positional_offsets.at(input_node_id);
      uint32_t end_position = positional_offsets.at(input_node_id + 1);
      std::copy(current_input.activations,
                current_input.activations + current_input.len,
                output_vector.activations + start_position);

      if (!current_input.isDense()) {
        assert(!output_vector.isDense());
        uint32_t neuron_index_offset = neuron_index_offsets.at(input_node_id);
        std::copy(current_input.active_neurons,
                  current_input.active_neurons + current_input.len,
                  output_vector.active_neurons + start_position);
        for (uint32_t output_position = start_position;
             output_position < end_position; output_position++) {
          output_vector.active_neurons[output_position] += neuron_index_offset;
        }
      }
    }
  }

  void backpropagateImpl(uint32_t vec_index) final {
    const auto& concatenated_nodes = _graph_state->inputs;
    const auto& positional_offsets =
        _batch_processing_state->positional_offsets;
    const auto& output_vector = getOutputVectorImpl(vec_index);

    for (uint32_t input_node_id = 0; input_node_id < concatenated_nodes.size();
         input_node_id++) {
      const auto& current_input_node = concatenated_nodes.at(input_node_id);
      BoltVector& current_input =
          current_input_node->getOutputVector(vec_index);
      uint32_t start_position = positional_offsets.at(input_node_id);
      uint32_t end_position = positional_offsets.at(input_node_id + 1);

      for (uint32_t output_position = start_position;
           output_position < end_position; output_position++) {
        current_input.gradients[output_position - start_position] +=
            output_vector.gradients[output_position];
      }
    }
  }

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

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    (void)detailed;
    const auto& inputs = _graph_state->inputs;
    summary << "(";
    for (uint32_t i = 0; i < inputs.size(); i++) {
      summary << inputs.at(i)->name();
      if (i != inputs.size() - 1) {
        summary << ", ";
      }
    }
    summary << ") -> " << name() << " (Concatenate)\n";
  }

  std::string type() const final { return "concat"; }

  static void verifyNoInputNodes(const std::vector<NodePtr>& nodes) {
    for (const auto& node : nodes) {
      if (node->isInputNode()) {
        throw exceptions::GraphCompilationFailure(
            "Cannot directly concatenate with an input node");
      }
    }
  }

  static bool concatenationHasSparseNode(const std::vector<NodePtr>& nodes) {
    return std::any_of(nodes.begin(), nodes.end(), nodeIsSparse);
  }

  static bool nodeIsSparse(const NodePtr& node) {
    return node->numNonzerosInOutput() < node->outputDim();
  }

  static std::vector<uint32_t> getPositionalOffsets(
      const std::vector<NodePtr>& nodes, bool use_sparsity) {
    std::vector<uint32_t> offsets = {0};
    uint64_t current_offset = 0;
    for (const auto& node : nodes) {
      current_offset +=
          use_sparsity ? node->numNonzerosInOutput() : node->outputDim();
      offsets.push_back(current_offset);
    }
    if (current_offset > std::numeric_limits<uint32_t>::max()) {
      throw exceptions::NodeStateMachineError(
          "Sum of input node dimensions must be less than UINT32_MAX: " +
          std::to_string(std::numeric_limits<uint32_t>::max()) + ", but was " +
          std::to_string(current_offset));
    }
    return offsets;
  }

  static BoltBatch generateBatch(
      bool use_sparsity, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& neuron_index_offsets, uint32_t batch_size) {
    BoltBatch batch = BoltBatch(/* dim= */ positional_offsets.back(),
                                /* batch_size= */ batch_size,
                                /* is_dense= */ !use_sparsity);
    if (use_sparsity) {
      fillSparseBatchWithConsecutiveIndices(
          batch, /* positional_offsets = */ positional_offsets,
          /* neuron_index_offsets = */ neuron_index_offsets);
    }
    return batch;
  }

  // This method prefills the active_neurons for the concatenated output,
  // assuming that each vector is dense. This has no impact on sparse parts
  // of the concatenation since they will override the preset active_neurons.
  static void fillSparseBatchWithConsecutiveIndices(
      BoltBatch& batch, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& neuron_index_offsets) {
    for (uint32_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
      for (uint32_t input_node_id = 0;
           input_node_id < positional_offsets.size() - 1; input_node_id++) {
        uint32_t start_position = positional_offsets.at(input_node_id);
        uint32_t end_position = positional_offsets.at(input_node_id + 1);
        uint32_t neuron_index_offset = neuron_index_offsets.at(input_node_id);
        std::iota(batch[vec_id].active_neurons + start_position,
                  batch[vec_id].active_neurons + end_position,
                  neuron_index_offset);
      }
    }
  }

  NodeState getState() const final {
    if (!_graph_state && !_compiled && !_batch_processing_state) {
      return NodeState::Constructed;
    }
    if (_graph_state && !_compiled && !_batch_processing_state) {
      return NodeState::PredecessorsSet;
    }
    if (_graph_state && _compiled && !_batch_processing_state) {
      return NodeState::Compiled;
    }
    if (_graph_state && _compiled && _batch_processing_state) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "ConcatenateNode is in an invalid internal state");
  }

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

CEREAL_REGISTER_TYPE(thirdai::bolt::ConcatenateNode)