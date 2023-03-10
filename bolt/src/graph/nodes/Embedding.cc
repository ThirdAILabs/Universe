#include "Embedding.h"
#include <cereal/archives/binary.hpp>

namespace thirdai::bolt {

EmbeddingNode::EmbeddingNode(uint64_t num_embedding_lookups,
                             uint64_t lookup_size,
                             uint64_t log_embedding_block_size,
                             const std::string& reduction,
                             std::optional<uint64_t> num_tokens_per_input)
    : _embedding_layer(nullptr),
      _config(EmbeddingLayerConfig(
          /* num_embedding_lookups= */ num_embedding_lookups,
          /* lookup_size= */ lookup_size,
          /* log_embedding_block_size= */ log_embedding_block_size,
          /* reduction= */ reduction,
          /* num_tokens_per_input= */ num_tokens_per_input)),
      _outputs(std::nullopt),
      _token_input(nullptr) {}

NodePtr EmbeddingNode::cloneForLayerSharing() {
  auto config = _config.value_or(_embedding_layer->getConfig());

  return std::shared_ptr<EmbeddingNode>(
      new EmbeddingNode(config.numEmbeddingLookups(), config.lookupSize(),
                        config.logEmbeddingBlockSize(),
                        config.reductionString(), config.numTokensPerInput()));
}
void EmbeddingNode::shareLayer(NodePtr& other) {
  auto other_emb = std::dynamic_pointer_cast<EmbeddingNode>(other);
  if (!other_emb) {
    throw std::invalid_argument(
        "Cannot share non-embedding node params with embedding node.");
  }

  NodeState node_state = getState();
  if (node_state == NodeState::Constructed ||
      node_state == NodeState::PredecessorsSet) {
    throw std::invalid_argument("Called copy() before compiling.");
  }

  NodeState other_node_state = other_emb->getState();
  if (other_node_state == NodeState::Constructed ||
      other_node_state == NodeState::PredecessorsSet) {
    throw std::invalid_argument("Tried to copy a precompiled layer.");
  }

  if (other_emb->outputDim() != outputDim()) {
    std::stringstream invalid_size;
    invalid_size << "incompatible embeddings sizes " << outputDim() << " vs. "
                 << other_emb->outputDim() << ".";
    throw std::invalid_argument(invalid_size.str());
  }

  _embedding_layer = other_emb->_embedding_layer;
}

uint32_t EmbeddingNode::outputDim() const {
  NodeState node_state = getState();
  if (node_state == NodeState::Constructed ||
      node_state == NodeState::PredecessorsSet) {
    return _config->getOutputDim();
  }
  return _embedding_layer->getOutputDim();
}

std::shared_ptr<EmbeddingNode> EmbeddingNode::addInput(InputPtr input) {
  if (getState() != NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "EmbeddingNodes have exactly one TokenInput node as input and the "
        "addInput function cannot be called more than once.");
  }

  _token_input = std::move(input);

  return shared_from_this();
}

Node::NodeState EmbeddingNode::getState() const {
  if (_token_input == nullptr && _embedding_layer == nullptr &&
      !_outputs.has_value()) {
    return NodeState::Constructed;
  }
  if (_token_input != nullptr && _embedding_layer == nullptr &&
      !_outputs.has_value()) {
    return NodeState::PredecessorsSet;
  }
  if (_token_input != nullptr && _embedding_layer != nullptr &&
      !_outputs.has_value()) {
    return NodeState::Compiled;
  }
  if (_token_input != nullptr && _embedding_layer != nullptr &&
      _outputs.has_value()) {
    return NodeState::PreparedForBatchProcessing;
  }
  throw exceptions::NodeStateMachineError(
      "EmbeddingNode is in an invalid internal state");
}

void EmbeddingNode::disableSparseParameterUpdates() {
  if (getState() != NodeState::Compiled &&
      getState() != NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "Cannot call disable_sparse_parameter_updates until the model "
        "containing the node is compiled.");
  }
  _embedding_layer->disableSparseParameterUpdates();
}

void EmbeddingNode::compileImpl() {
  assert(_config.has_value());
  _embedding_layer = std::make_shared<EmbeddingLayer>(_config.value());
  _config = std::nullopt;
}

void EmbeddingNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                                  bool use_sparsity) {
  (void)use_sparsity;

  _outputs = _embedding_layer->createBatchState(batch_size);
}

void EmbeddingNode::forwardImpl(uint32_t vec_index, const BoltVector* labels) {
  (void)labels;

  _embedding_layer->forward(
      /* tokens= */ _token_input->getOutputVector(vec_index),
      /* output= */ (*_outputs)[vec_index]);
}

void EmbeddingNode::backpropagateImpl(uint32_t vec_index) {
  _embedding_layer->backpropagate(
      /* tokens= */ _token_input->getOutputVector(vec_index),
      /* output= */ (*_outputs)[vec_index]);
}

void EmbeddingNode::updateParametersImpl(float learning_rate,
                                         uint32_t batch_cnt) {
  _embedding_layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2,
                                     EPS);
}

void EmbeddingNode::summarizeImpl(std::stringstream& summary,
                                  bool detailed) const {
  (void)detailed;
  summary << _token_input->name() << " -> " << name() << ": (Embedding):";
  _embedding_layer->buildLayerSummary(summary);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::EmbeddingNode)
