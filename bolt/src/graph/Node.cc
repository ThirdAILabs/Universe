#include "Node.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/string.hpp>
#include <stdexcept>

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

  if (batch_size > 0 && getOutputVector(0).len == 0) {
    throw std::invalid_argument(
        "Node: '" + name() +
        "' allocated with output dimension=0. Please check that the the "
        "dimensions of nodes are nonzero, and that the dimension will be "
        "nonzero after applying sparsity.");
  }
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

template <class Archive>
void Node::serialize(Archive& archive) {
  archive(_name);
}

template void Node::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);
template void Node::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void Node::serialize<cereal::PortableBinaryInputArchive>(
    cereal::PortableBinaryInputArchive&);
template void Node::serialize<cereal::PortableBinaryOutputArchive>(
    cereal::PortableBinaryOutputArchive&);

}  // namespace thirdai::bolt
CEREAL_REGISTER_TYPE(thirdai::bolt::Node)
