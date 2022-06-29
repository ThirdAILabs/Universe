#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::bolt {

class ConcatenatedNode final
    : public Node,
      public std::enable_shared_from_this<ConcatenatedNode> {
 public:
  ConcatenatedNode(){};

  void initializeParameters() final {}

  void forward(uint32_t vec_index, const BoltVector* labels) final {
    // We currently do not allow a concatenation layer to be the last
    // layer in the graph.
    // TODO(josh/nick): Add support for n sets of outputs, and if users want
    // a concatenation layer as the last layer they can split the labels
    assert(labels == nullptr);
    (void)labels;
    assert(prepared_for_batch_processing() && predecessors_set());

    const auto& concatenated_nodes = _graph_state->concatenated_nodes;
    const auto& positional_offsets =
        _batch_processing_state->positional_offsets;
    const auto& label_offsets = _graph_state->label_offsets;
    const auto& output_vector = getOutputVector(vec_index);

    for (uint32_t concat_id = 0; concat_id < concatenated_nodes.size();
         concat_id++) {
      const auto& node = concatenated_nodes.at(concat_id);
      auto& current_concat_input = node->getOutputVector(vec_index);
      uint32_t start_offset = positional_offsets.at(concat_id);
      uint32_t end_offset = positional_offsets.at(concat_id + 1);
      uint32_t label_starting_offset = label_offsets.at(concat_id);
      for (uint32_t index = start_offset; index < end_offset; index++) {
        output_vector.activations[index] =
            current_concat_input.activations[index - start_offset];
        if (!current_concat_input.isDense()) {
          output_vector.active_neurons[index] =
              current_concat_input.active_neurons[index - start_offset] +
              label_starting_offset;
        }
      }
    }
  }

  void backpropagate(uint32_t vec_index) final {
    assert(prepared_for_batch_processing() && predecessors_set());

    const auto& concatenated_nodes = _graph_state->concatenated_nodes;
    const auto& positional_offsets =
        _batch_processing_state->positional_offsets;
    const auto& output_vector = getOutputVector(vec_index);

    for (uint32_t concat_id = 0; concat_id < concatenated_nodes.size();
         concat_id++) {
      const auto& node = concatenated_nodes.at(concat_id);
      auto& current_concat_input = node->getOutputVector(vec_index);
      uint32_t start_offset = positional_offsets.at(concat_id);
      uint32_t end_offset = positional_offsets.at(concat_id + 1);
      for (uint32_t index = start_offset; index < end_offset; index++) {
        current_concat_input.gradients[index - start_offset] =
            output_vector.gradients[index];
      }
    }
  }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    (void)learning_rate;
    (void)batch_cnt;
    // NOOP because a concatenation layer has no parameters
  }

  BoltVector& getOutputVector(uint32_t vec_index) final {
    assert(prepared_for_batch_processing());
    return _batch_processing_state->outputs[vec_index];
  }

  std::shared_ptr<ConcatenatedNode> setConcatenatedNodes(
      const std::vector<NodePtr>& nodes) {
    if (predecessors_set()) {
      throw std::logic_error(
          "Have already set the incoming concatenated nodes for this "
          "concatenation layer");
    }
    if (nodes.empty()) {
      throw std::invalid_argument(
          "Must concatenate at least one node, found 0");
    }

    verifyNotConcatenatingInputNode(nodes);

    std::vector<uint32_t> label_offsets = {0};
    uint32_t output_dim = 0;
    for (const auto& node : nodes) {
      output_dim += node->outputDim();
      label_offsets.push_back(output_dim);
    }
    _graph_state = {/* concatenated_nodes = */ nodes,
                    /* label_offsets = */ label_offsets,
                    /* output_dim = */ output_dim};

    return shared_from_this();
  }

  uint32_t outputDim() const final {
    if (!predecessors_set()) {
      throw std::logic_error(
          "Cannot get the output dim for this concatenation layer because the "
          "incoming concatenated nodes have not been set yet");
    }
    return _graph_state->output_dim;
  }

  uint32_t numNonzerosInOutput() const final {
    if (!prepared_for_batch_processing()) {
      throw std::logic_error(
          "Cannot get the number of nonzeros for this concatenation layer "
          "because the node is not prepared for batch processing");
    }
    return _batch_processing_state->num_nonzeros;
  }

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) final {
    if (!predecessors_set()) {
      throw std::logic_error(
          "The preceeding nodes to this concatenation layer "
          " must be set before preparing for batch processing.");
    }
    const auto& concatenated_nodes = _graph_state->concatenated_nodes;

    bool concatenation_sparse =
        concatenationHasSparseNode(concatenated_nodes) && use_sparsity;
    std::vector<uint32_t> positional_offsets = getPositionalOffsets(
        concatenated_nodes, /* use_sparsity = */ concatenation_sparse);
    BoltBatch new_concateated_batch = generateBatch(
        /* use_sparsity = */ concatenation_sparse,
        /* positional_offsets = */ positional_offsets,
        /* label_offsets = */ _graph_state->label_offsets,
        /* batch_size = */ batch_size);

    _batch_processing_state = {
        /* num_nonzeros = */ positional_offsets.back(),
        /* positional_offsets = */ std::move(positional_offsets),
        /* outputs = */ std::move(new_concateated_batch)};
  }

  std::vector<NodePtr> getPredecessors() const final {
    if (!predecessors_set()) {
      throw std::logic_error(
          "Cannot get the predecessors for this concatenation layer because "
          "they have not been set yet");
    }
    return _graph_state->concatenated_nodes;
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    if (!predecessors_set()) {
      throw std::logic_error(
          "getInternalFullyConnectedLayers method should not be called before "
          "predecessors have been set");
    }
    return {};
  }

  bool isInputNode() const final { return false; }

 private:
  static void verifyNotConcatenatingInputNode(
      const std::vector<NodePtr>& nodes) {
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
    std::vector<uint32_t> new_offsets = {0};
    uint64_t current_offset = 0;
    for (const auto& node : nodes) {
      current_offset +=
          use_sparsity ? node->numNonzerosInOutput() : node->outputDim();
      new_offsets.push_back(current_offset);
    }
    if (current_offset > UINT32_MAX) {
      throw std::logic_error(
          "Sum of input node dimensions must be less than UINT32_MAX: " +
          std::to_string(UINT32_MAX) + ", but was " +
          std::to_string(current_offset));
    }
    return new_offsets;
  }

  static BoltBatch generateBatch(
      bool use_sparsity, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& label_offsets, uint32_t batch_size) {
    BoltBatch batch = BoltBatch(/* dim= */ positional_offsets.back(),
                                /* batch_size= */ batch_size,
                                /* is_dense= */ !use_sparsity);
    if (use_sparsity) {
      fillSparseBatchWithConsecutiveIndices(
          batch, /* positional_offsets = */ positional_offsets,
          /* label_offsets = */ label_offsets);
    }
    return batch;
  }

  static void fillSparseBatchWithConsecutiveIndices(
      BoltBatch& batch, const std::vector<uint32_t>& positional_offsets,
      const std::vector<uint32_t>& label_offsets) {
    for (uint32_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
      for (uint32_t concat_id = 0; concat_id < positional_offsets.size() - 1;
           concat_id++) {
        uint32_t start_offset = positional_offsets.at(concat_id);
        uint32_t end_offset = positional_offsets.at(concat_id + 1);
        uint32_t starting_label = label_offsets.at(concat_id);
        for (uint32_t offset = start_offset; offset < end_offset; offset++) {
          batch[vec_id].active_neurons[offset] =
              starting_label + (offset - start_offset);
        }
      }
    }
  }

  bool predecessors_set() const { return _graph_state.has_value(); }

  bool prepared_for_batch_processing() const {
    return _batch_processing_state.has_value();
  }

  struct GraphState {
    std::vector<NodePtr> concatenated_nodes;
    // The ith element is the label offset for the ith concat.
    // Contains len(concatenated_nodes) + 1 number of elements, as we use the
    // pattern where the last item would be the label start for the "next"
    // concat if there was one.
    std::vector<uint32_t> label_offsets;
    // Also equals the last element of label_offsets
    uint32_t output_dim;
  };
  struct BatchProcessingState {
    uint32_t num_nonzeros;
    // The ith element is the positional offset for the ith concatenated output.
    // We use the same pattern as for label_offsets above, so there are
    // len(concatenated_nodes) + 1 number of elements
    std::vector<uint32_t> positional_offsets;
    // Each vector here has dimension num_nonzeros
    BoltBatch outputs;
  };

  std::optional<GraphState> _graph_state;
  std::optional<BatchProcessingState> _batch_processing_state;
};

}  // namespace thirdai::bolt
