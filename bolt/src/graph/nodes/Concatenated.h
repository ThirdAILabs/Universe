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
    assert(prepared_for_batch_processing());

    const auto& concatenated_nodes = _graph_state->concatenated_nodes;
    const auto& offsets = _batch_processing_state->offsets;
    const auto& output_vector = getOutputVector(vec_index);

    for (uint32_t concat_id = 0; concat_id < concatenated_nodes.size();
         concat_id++) {
      const auto& node = concatenated_nodes.at(concat_id);
      auto& current_concatenated_input = node->getOutputVector(vec_index);
      uint32_t start_offset = offsets.at(concat_id);
      uint32_t end_offset = offsets.at(concat_id);
      for (uint32_t index = start_offset; index < end_offset; index++) {
        output_vector.activations[index] =
            current_concatenated_input.activations[index - start_offset];
        if (!current_concatenated_input.isDense()) {
          output_vector.active_neurons[index] =
              current_concatenated_input.active_neurons[index - start_offset];
        }
      }
    }
  }

  void backpropagate(uint32_t vec_index) final {
    auto& output_vector = getOutputVector(vec_index);
    for (uint32_t concat_id = 0; concat_id < _concatenated_nodes.size();
         concat_id++) {
      auto& node = _concatenated_nodes.at(concat_id);
      auto& current_concatenated_input = node->getOutputVector(vec_index);
      uint32_t start_offset = _offsets.at(concat_id);
      uint32_t end_offset = _offsets.at(concat_id);
      for (uint32_t index = start_offset; index < end_offset; index++) {
        current_concatenated_input.gradients[index - start_offset] =
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
    if (input_nodes_set()) {
      throw std::logic_error(
          "Have already set the incoming concatenated nodes for this "
          "concatenation layer");
    }
    if (nodes.empty()) {
      throw std::invalid_argument(
          "Must concatenate at least one node, found 0");
    }

    verifyNotConcatenatingInputNode(nodes);

    uint32_t output_dim = 0;
    for (const auto& node : nodes) {
      output_dim += node->outputDim();
    }
    _graph_state = {/* concatenated_nodes = */ nodes,
                    /* output_dim = */ output_dim};

    return shared_from_this();
  }

  uint32_t outputDim() const final {
    if (!input_nodes_set()) {
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
    if (!input_nodes_set()) {
      throw std::logic_error(
          "The preceeding nodes to this concatenation layer "
          " must be set before preparing for batch processing.");
    }
    bool concatenation_sparse =
        concatenationHasSparseNode(_concatenated_nodes) && use_sparsity;
    std::vector<uint32_t> new_offsets =
        getOffsets(_concatenated_nodes, concatenation_sparse);
    BoltBatch new_concateated_batch = generateBatch(
        /* is_sparse = */ concatenation_sparse, /* offsets = */ new_offsets,
        /* batch_size = */ batch_size);

    _offsets = new_offsets;
    _num_nonzeros = new_offsets.back();
    _outputs = std::move(new_concateated_batch);
  }

  std::vector<NodePtr> getPredecessors() const final {
    if (!input_nodes_set()) {
      throw std::logic_error(
          "Cannot get the predecessors for this concatenation layer because "
          "they have not been set yet");
    }
    return _graph_state->concatenated_nodes;
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    if (!input_nodes_set()) {
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
    return node->numNonzerosInOutput().value() < node->outputDim();
  }

  static std::vector<uint32_t> getOffsets(const std::vector<NodePtr>& nodes,
                                          bool concatenation_is_sparse) {
    std::vector<uint32_t> new_offsets = {0};
    uint64_t current_offset = 0;
    for (const auto& node : nodes) {
      uint32_t node_dim = concatenation_is_sparse ? node->numNonzerosInOutput()
                                                  : node->outputDim();
      current_offset += node_dim;
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

  static BoltBatch generateBatch(bool is_sparse,
                                 const std::vector<uint32_t>& offsets,
                                 uint32_t batch_size) {
    BoltBatch batch =
        BoltBatch(/* dim= */ offsets.back(), /* batch_size= */ batch_size,
                  /* is_dense= */ !is_sparse);
    if (is_sparse) {
      fillSparseBatchWithConsecutiveIndices(batch, offsets);
    }
    return batch;
  }

  static void fillSparseBatchWithConsecutiveIndices(
      BoltBatch& batch, const std::vector<uint32_t>& offsets) {
    for (uint32_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
      for (uint32_t concat_id = 0; concat_id < offsets.size() - 1;
           concat_id++) {
        uint32_t start_offset = offsets.at(concat_id);
        uint32_t end_offset = offsets.at(concat_id + 1);
        for (uint32_t offset = start_offset; offset < end_offset; offset++) {
          batch[vec_id].active_neurons[offset] = offset - start_offset;
        }
      }
    }
  }

  bool input_nodes_set() const { return _graph_state.has_value(); }

  bool prepared_for_batch_processing() const {
    return _batch_processing_state.has_value();
  }

  struct GraphState {
    std::vector<NodePtr> concatenated_nodes;
    uint32_t output_dim;
  };
  struct BatchProcessingState {
    uint32_t num_nonzeros;
    std::vector<uint32_t> offsets;
    BoltBatch outputs;
  };

  std::optional<GraphState> _graph_state;
  std::optional<BatchProcessingState> _batch_processing_state;
};

}  // namespace thirdai::bolt
