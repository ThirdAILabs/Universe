#include "Concatenate.h"

namespace thirdai::bolt {

std::shared_ptr<ConcatenateNode> ConcatenateNode::setConcatenatedNodes(
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

uint32_t ConcatenateNode::outputDim() const {
  if (getState() == NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "Cannot get the output dim for this concatenation layer because the "
        "incoming concatenated nodes have not been set yet");
  }
  return _graph_state->concatenated_dense_dim;
}

void ConcatenateNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                                    bool use_sparsity) {
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

void ConcatenateNode::forwardImpl(uint32_t vec_index,
                                  const BoltVector* labels) {
  // We currently do not allow a concatenation layer to be the last
  // layer in the graph.
  // TODO(josh/nick): Add support for n sets of outputs, and if users want
  // a concatenation layer as the last layer they can split the labels
  assert(labels == nullptr);
  (void)labels;

  const BoltVector& output_vector = getOutputVectorImpl(vec_index);

  const auto& concatenated_nodes = _graph_state->inputs;
  const auto& positional_offsets = _batch_processing_state->positional_offsets;
  const auto& neuron_index_offsets = _graph_state->neuron_index_offsets;

  for (uint32_t input_node_id = 0; input_node_id < concatenated_nodes.size();
       input_node_id++) {
    const auto& current_input_node = concatenated_nodes.at(input_node_id);
    BoltVector& current_input = current_input_node->getOutputVector(vec_index);
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

void ConcatenateNode::backpropagateImpl(uint32_t vec_index) {
  const auto& concatenated_nodes = _graph_state->inputs;
  const auto& positional_offsets = _batch_processing_state->positional_offsets;
  const auto& output_vector = getOutputVectorImpl(vec_index);

  for (uint32_t input_node_id = 0; input_node_id < concatenated_nodes.size();
       input_node_id++) {
    const auto& current_input_node = concatenated_nodes.at(input_node_id);
    BoltVector& current_input = current_input_node->getOutputVector(vec_index);
    uint32_t start_position = positional_offsets.at(input_node_id);
    uint32_t end_position = positional_offsets.at(input_node_id + 1);

    for (uint32_t output_position = start_position;
         output_position < end_position; output_position++) {
      current_input.gradients[output_position - start_position] +=
          output_vector.gradients[output_position];
    }
  }
}

void ConcatenateNode::summarizeImpl(std::stringstream& summary,
                                    bool detailed) const {
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

void ConcatenateNode::verifyNoInputNodes(const std::vector<NodePtr>& nodes) {
  for (const auto& node : nodes) {
    if (node->isInputNode()) {
      throw exceptions::GraphCompilationFailure(
          "Cannot directly concatenate with an input node");
    }
  }
}

std::vector<uint32_t> ConcatenateNode::getPositionalOffsets(
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

BoltBatch ConcatenateNode::generateBatch(
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

void ConcatenateNode::fillSparseBatchWithConsecutiveIndices(
    BoltBatch& batch, const std::vector<uint32_t>& positional_offsets,
    const std::vector<uint32_t>& neuron_index_offsets) {
  for (const auto& vec : batch) {
    for (uint32_t input_node_id = 0;
         input_node_id < positional_offsets.size() - 1; input_node_id++) {
      uint32_t start_position = positional_offsets.at(input_node_id);
      uint32_t end_position = positional_offsets.at(input_node_id + 1);
      uint32_t neuron_index_offset = neuron_index_offsets.at(input_node_id);
      std::iota(vec.active_neurons + start_position,
                vec.active_neurons + end_position, neuron_index_offset);
    }
  }
}

Node::NodeState ConcatenateNode::getState() const {
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

}  // namespace thirdai::bolt