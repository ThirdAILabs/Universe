#include "Conv.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/nodes/Input3D.h>
#include <bolt/src/layers/LayerConfig.h>
#include <tuple>

namespace thirdai::bolt {

ConvNode::ConvNode(uint64_t num_filters, const std::string& activation,
                   std::pair<uint32_t, uint32_t> kernel_size,
                   std::pair<uint32_t, uint32_t> next_kernel_size)
    : _layer(nullptr),
      _config(ConvLayerConfig(num_filters, activation, std::move(kernel_size),
                              next_kernel_size)),
      _predecessor(nullptr) {}

ConvNode::ConvNode(uint64_t num_filters, float sparsity,
                   const std::string& activation,
                   std::pair<uint32_t, uint32_t> kernel_size,
                   std::pair<uint32_t, uint32_t> next_kernel_size)
    : _layer(nullptr),
      _config(ConvLayerConfig(num_filters, sparsity, activation,
                              std::move(kernel_size), next_kernel_size)),
      _predecessor(nullptr) {}

ConvNode::ConvNode(uint64_t num_filters, float sparsity,
                   const std::string& activation,
                   std::pair<uint32_t, uint32_t> kernel_size,
                   std::pair<uint32_t, uint32_t> next_kernel_size,
                   SamplingConfigPtr sampling_config)
    : _layer(nullptr),
      _config(ConvLayerConfig(num_filters, sparsity, activation,
                              std::move(kernel_size), next_kernel_size,
                              std::move(sampling_config))),
      _predecessor(nullptr) {}

std::shared_ptr<ConvNode> ConvNode::makeDense(
    uint32_t num_filters, const std::string& activation,
    std::pair<uint32_t, uint32_t> kernel_size,
    std::pair<uint32_t, uint32_t> next_kernel_size) {
  return std::shared_ptr<ConvNode>(
      new ConvNode(num_filters, activation, kernel_size, next_kernel_size));
}

std::shared_ptr<ConvNode> ConvNode::makeAutotuned(
    uint32_t num_filters, float sparsity, const std::string& activation,
    std::pair<uint32_t, uint32_t> kernel_size,
    std::pair<uint32_t, uint32_t> next_kernel_size) {
  return std::shared_ptr<ConvNode>(new ConvNode(
      num_filters, sparsity, activation, kernel_size, next_kernel_size));
}

std::shared_ptr<ConvNode> ConvNode::make(
    uint32_t num_filters, float sparsity, const std::string& activation,
    std::pair<uint32_t, uint32_t> kernel_size,
    std::pair<uint32_t, uint32_t> next_kernel_size,
    SamplingConfigPtr sampling_config) {
  return std::shared_ptr<ConvNode>(
      new ConvNode(num_filters, sparsity, activation, kernel_size,
                   next_kernel_size, std::move(sampling_config)));
}

std::shared_ptr<ConvNode> ConvNode::addPredecessor(NodePtr node) {
  if (getState() != NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "ConvNode expected to have exactly one predecessor, and "
        "addPredecessor cannot be called twice.");
  }

  if (!std::dynamic_pointer_cast<ConvNode>(node) &&
      !std::dynamic_pointer_cast<Input3D>(node)) {
    throw std::invalid_argument(
        "Previous node must have a 3D output (currently must be Conv or "
        "Input3D).");
  }

  _predecessor = std::move(node);

  return shared_from_this();
}

uint32_t ConvNode::outputDim() const {
  NodeState node_state = getState();
  if (node_state == NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "Cannot calculate output dimension of a ConvNode before setting the "
        "predecessor.");
  }
  if (node_state == NodeState::PredecessorsSet) {
    auto [height, width, depth] = getPredecessorOutputDim();
    return (height / (*_config).kernel_size.first) *
           (width / (*_config).kernel_size.second) * (*_config).num_filters;
  }
  return _layer->getDim();
}

void ConvNode::compileImpl() {
  assert(_config.has_value());

  auto [height, width, depth] = getPredecessorOutputDim();

  _layer = std::make_shared<ConvLayer>(_config.value(), /* height= */ height,
                                       /* width= */ width, /* depth= */ depth,
                                       /* prev_sparsity= */ 1);

  _config = std::nullopt;
}

void ConvNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                             bool use_sparsity) {
  _outputs =
      _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
}

void ConvNode::forwardImpl(uint32_t vec_index, const BoltVector* labels) {
  _layer->forward(_predecessor->getOutputVector(vec_index),
                  this->getOutputVectorImpl(vec_index), labels);
}

void ConvNode::backpropagateImpl(uint32_t vec_index) {
  if (_predecessor->isInputNode()) {
    _layer->backpropagateInputLayer(_predecessor->getOutputVector(vec_index),
                                    this->getOutputVectorImpl(vec_index));
  } else {
    _layer->backpropagate(_predecessor->getOutputVector(vec_index),
                          this->getOutputVectorImpl(vec_index));
  }
}

void ConvNode::updateParametersImpl(float learning_rate, uint32_t batch_cnt) {
  // TODO(Nicholas): Abstract away these constants
  _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
}

void ConvNode::summarizeImpl(std::stringstream& summary, bool detailed) const {
  (void)detailed;
  summary << _predecessor->name() << " -> " << name() << " (Conv): ";
  _layer->buildLayerSummary(summary);
}

Node::NodeState ConvNode::getState() const {
  if (_predecessor == nullptr && _layer == nullptr && !_outputs.has_value()) {
    return NodeState::Constructed;
  }
  if (_predecessor != nullptr && _layer == nullptr && !_outputs.has_value()) {
    return NodeState::PredecessorsSet;
  }
  if (_predecessor != nullptr && _layer != nullptr && !_outputs.has_value()) {
    return NodeState::Compiled;
  }
  if (_predecessor != nullptr && _layer != nullptr && _outputs.has_value()) {
    return NodeState::PreparedForBatchProcessing;
  }
  throw exceptions::NodeStateMachineError(
      "ConvNode is in an invalid internal state");
}

std::tuple<uint32_t, uint32_t, uint32_t> ConvNode::getPredecessorOutputDim()
    const {
  if (_predecessor == nullptr) {
    throw std::invalid_argument(
        "Cannot get the output dim of predecessor since it is not set yet.");
  }

  if (auto conv_node = std::dynamic_pointer_cast<ConvNode>(_predecessor)) {
    return conv_node->getOutputDim3D();
  }

  if (auto input_3d_node = std::dynamic_pointer_cast<Input3D>(_predecessor)) {
    return input_3d_node->getOutputDim3D();
  }

  throw std::invalid_argument(
      "Predecessor of ConvNode is not ConvNode or Input3D.");
}

uint32_t ConvNode::getSparsity() const {
  if (_config.has_value()) {
    return _config->sparsity;
  }
  return _layer->getSparsity();
}

std::tuple<uint32_t, uint32_t, uint32_t> ConvNode::getOutputDim3D() const {
  if (_layer == nullptr) {
    throw std::invalid_argument(
        "Not compiled. Cannot access output dim without compiling.");
  }
  return _layer->getOutputDim3D();
}

template <class Archive>
void ConvNode::serialize(Archive& archive) {
  archive(cereal::base_class<Node>(this), _layer, _config, _predecessor);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::ConvNode)