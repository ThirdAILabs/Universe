#include "Node.h"

namespace thirdai::bolt {

void Node::compile(LayerNameManager& name_manager) {
  if (getState() == NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "Cannot call compile before setting predecessor(s) of this Node.");
  }
  if (getState() == NodeState::Compiled ||
      getState() == NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError("Cannot call compile twice.");
  }
  _name = name_manager.registerNodeAndGetName(/* node_type = */ type());
  compileImpl();
}

inline void Node::forward(uint32_t vec_index, const BoltVector* labels) {
  assert(getState() == NodeState::PreparedForBatchProcessing);
  forwardImpl(vec_index, labels);
}

inline void Node::backpropagate(uint32_t vec_index) {
  assert(getState() == NodeState::PreparedForBatchProcessing);
  backpropagateImpl(vec_index);
}

inline void Node::updateParameters(float learning_rate, uint32_t batch_cnt) {
  assert(getState() == NodeState::PreparedForBatchProcessing);
  updateParametersImpl(learning_rate, batch_cnt);
}

inline BoltVector& Node::getOutputVector(uint32_t vec_index) {
  assert(getState() == NodeState::PreparedForBatchProcessing);
  return getOutputVectorImpl(vec_index);
}

uint32_t Node::numNonzerosInOutput() const {
  if (getState() != NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "Must call prepareForBatchProcessing before calling "
        "numNonzerosInOutput.");
  }

  return numNonzerosInOutputImpl();
}

void Node::prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) {
  if (getState() == NodeState::Constructed ||
      getState() == NodeState::PredecessorsSet) {
    throw exceptions::NodeStateMachineError(
        "Cannot call prepareForBatchProcessing before calling compile.");
  }

  if (getState() == NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "Cannot call prepareForBatchProcessing consecutively (must call "
        "cleanupAfterBatchProcessing in between).");
  }

  prepareForBatchProcessingImpl(batch_size, use_sparsity);
}

void Node::cleanupAfterBatchProcessing() {
  if (getState() != Node::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "Can only call cleanupAfterBatchProcessing after "
        "prepareForBatchProcessing.");
  }

  cleanupAfterBatchProcessingImpl();
}

std::vector<NodePtr> Node::getPredecessors() const {
  if (getState() == NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "Cannot get the predecessors for this layer because "
        "they have not been set yet");
  }
  return getPredecessorsImpl();
}

std::vector<std::shared_ptr<FullyConnectedLayer>>
Node::getInternalFullyConnectedLayers() {
  if (getState() == NodeState::Constructed ||
      getState() == NodeState::PredecessorsSet) {
    throw exceptions::NodeStateMachineError(
        "Cannot call getInternalFullyConnectedLayers before "
        "calling compile.");
  }
  return getInternalFullyConnectedLayersImpl();
}

void Node::summarize(std::stringstream& summary, bool detailed) const {
  if (getState() == NodeState::Constructed ||
      getState() == NodeState::PredecessorsSet) {
    throw exceptions::NodeStateMachineError(
        "Can only summarize a node after compiling");
  }
  summarizeImpl(summary, detailed);
}

const std::string& Node::name() const {
  if (getState() == NodeState::Constructed ||
      getState() == NodeState::PredecessorsSet) {
    throw exceptions::NodeStateMachineError(
        "Can only get the name of a node after compiling");
  }
  return *_name;
}

}  // namespace thirdai::bolt