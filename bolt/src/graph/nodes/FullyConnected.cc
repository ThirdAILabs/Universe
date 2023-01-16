#include "FullyConnected.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/Node.h>

namespace thirdai::bolt {

std::shared_ptr<FullyConnectedNode> FullyConnectedNode::makeDense(
    uint32_t dim, const std::string& activation) {
  return std::shared_ptr<FullyConnectedNode>(
      new FullyConnectedNode(dim, activation));
}

std::shared_ptr<FullyConnectedNode> FullyConnectedNode::makeAutotuned(
    uint32_t dim, float sparsity, const std::string& activation) {
  return std::shared_ptr<FullyConnectedNode>(
      new FullyConnectedNode(dim, sparsity, activation));
}

std::shared_ptr<FullyConnectedNode> FullyConnectedNode::make(
    uint32_t dim, float sparsity, const std::string& activation,
    SamplingConfigPtr sampling_config) {
  return std::shared_ptr<FullyConnectedNode>(new FullyConnectedNode(
      dim, sparsity, activation, std::move(sampling_config)));
}

std::shared_ptr<FullyConnectedNode>
FullyConnectedNode::makeExplicitSamplingConfig(uint32_t dim, float sparsity,
                                               const std::string& activation,
                                               uint32_t num_tables,
                                               uint32_t hashes_per_table,
                                               uint32_t reservoir_size) {
  auto sampling_config = std::make_shared<DWTASamplingConfig>(
      num_tables, hashes_per_table, reservoir_size);
  return make(dim, sparsity, activation, sampling_config);
}

FullyConnectedNode::FullyConnectedNode(uint64_t dim,
                                       const std::string& activation)
    : _layer(nullptr),
      _config(FullyConnectedLayerConfig(dim, activation)),
      _predecessor(nullptr) {}

FullyConnectedNode::FullyConnectedNode(uint64_t dim, float sparsity,
                                       const std::string& activation)
    : _layer(nullptr),
      _config(FullyConnectedLayerConfig(dim, sparsity, activation)),
      _predecessor(nullptr) {}

FullyConnectedNode::FullyConnectedNode(uint64_t dim, float sparsity,
                                       const std::string& activation,
                                       SamplingConfigPtr sampling_config)
    : _layer(nullptr),
      _config(FullyConnectedLayerConfig(dim, sparsity, activation,
                                        std::move(sampling_config))),
      _predecessor(nullptr) {}

std::shared_ptr<FullyConnectedNode> FullyConnectedNode::addPredecessor(
    NodePtr node) {
  if (getState() != NodeState::Constructed) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode expected to have exactly one predecessor, and "
        "addPredecessor cannot be called twice.");
  }
  _predecessor = std::move(node);

  return shared_from_this();
}

uint32_t FullyConnectedNode::outputDim() const {
  NodeState node_state = getState();
  if (node_state == NodeState::Constructed ||
      node_state == NodeState::PredecessorsSet) {
    return _config->getDim();
  }
  return _layer->getDim();
}

bool FullyConnectedNode::isInputNode() const { return false; }

void FullyConnectedNode::initOptimizer() { _layer->initOptimizer(); }

ActivationFunction FullyConnectedNode::getActivationFunction() const {
  NodeState node_state = getState();
  if (node_state == NodeState::Constructed ||
      node_state == NodeState::PredecessorsSet) {
    return _config->getActFunc();
  }
  return _layer->getActivationFunction();
}

void FullyConnectedNode::saveParameters(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(*_layer);
}

void FullyConnectedNode::loadParameters(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
  auto loaded_parameters = std::make_shared<FullyConnectedLayer>();
  iarchive(*loaded_parameters);

  if (loaded_parameters->getDim() != _layer->getDim()) {
    std::stringstream error_msg;
    error_msg << "Cannot load parameters from FullyConnected layer with dim="
              << loaded_parameters->getDim()
              << "into FullyConnected layer with dim=" << _layer->getDim()
              << ".";
    throw std::logic_error(error_msg.str());
  }
  if (loaded_parameters->getInputDim() != _layer->getInputDim()) {
    std::stringstream error_msg;
    error_msg
        << "Cannot load parameters from FullyConnected layer with input_dim="
        << loaded_parameters->getInputDim()
        << "into FullyConnected layer with input_dim=" << _layer->getInputDim()
        << ".";
    throw std::logic_error(error_msg.str());
  }

  if (loaded_parameters->getActivationFunction() !=
      _layer->getActivationFunction()) {
    std::stringstream error_msg;
    error_msg << "Cannot load parameters from FullyConnected layer with "
                 "activation ="
              << activationFunctionToStr(
                     loaded_parameters->getActivationFunction())
              << "into FullyConnected layer with activation="
              << activationFunctionToStr(_layer->getActivationFunction())
              << ".";
    throw std::logic_error(error_msg.str());
  }

  _layer = loaded_parameters;
}

float FullyConnectedNode::getSparsity() {
  NodeState node_state = getState();
  if (node_state == NodeState::Constructed ||
      node_state == NodeState::PredecessorsSet) {
    return _config->getSparsity();
  }
  return _layer->getSparsity();
}

std::shared_ptr<FullyConnectedNode> FullyConnectedNode::setSparsity(
    float sparsity) {
  if (getState() != NodeState::Compiled &&
      getState() != NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode must be in a compiled state to call "
        "setSparsity");
  }
  _layer->setSparsity(sparsity);
  return shared_from_this();
}

float* FullyConnectedNode::getWeightsPtr() {
  if (getState() != NodeState::PreparedForBatchProcessing &&
      getState() != NodeState::Compiled) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode must be in a compiled state to call "
        "getWeightsPtr.");
  }
  return _layer->getWeightsPtr();
}

float* FullyConnectedNode::getBiasesPtr() {
  if (getState() != NodeState::PreparedForBatchProcessing &&
      getState() != NodeState::Compiled) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode must be in a compiled state to call "
        "getBiasesPtr.");
  }
  return _layer->getBiasesPtr();
}

float* FullyConnectedNode::getWeightGradientsPtr() {
  if (getState() != NodeState::PreparedForBatchProcessing &&
      getState() != NodeState::Compiled) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode must be in a compiled state to call "
        "getWeightGradientsPtr.");
  }
  return _layer->getWeightGradientsPtr();
}

float* FullyConnectedNode::getBiasGradientsPtr() {
  if (getState() != NodeState::PreparedForBatchProcessing &&
      getState() != NodeState::Compiled) {
    throw exceptions::NodeStateMachineError(
        "FullyConnectedNode must be in a compiled state to call "
        "getBiasGradientsPtr.");
  }
  return _layer->getBiasGradientsPtr();
}

void FullyConnectedNode::disableSparseParameterUpdates() {
  if (getState() != NodeState::Compiled &&
      getState() != NodeState::PreparedForBatchProcessing) {
    throw exceptions::NodeStateMachineError(
        "Cannot call disable_sparse_parameter_updates until the model "
        "containing the node is compiled.");
  }
  _layer->disableSparseParameterUpdates();
}

void FullyConnectedNode::compileImpl() {
  assert(_config.has_value());
  _layer = std::make_shared<FullyConnectedLayer>(_config.value(),
                                                 _predecessor->outputDim());
  _config = std::nullopt;
}

std::vector<std::shared_ptr<FullyConnectedLayer>>
FullyConnectedNode::getInternalFullyConnectedLayersImpl() const {
  return {_layer};
}

void FullyConnectedNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                                       bool use_sparsity) {
  // TODO(Nicholas): rename createBatchState
  _outputs =
      _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
}

uint32_t FullyConnectedNode::numNonzerosInOutputImpl() const {
  return (*_outputs)[0].len;
}

void FullyConnectedNode::forwardImpl(uint32_t vec_index,
                                     const BoltVector* labels) {
  _layer->forward(_predecessor->getOutputVector(vec_index),
                  this->getOutputVectorImpl(vec_index), labels);
}

void FullyConnectedNode::backpropagateImpl(uint32_t vec_index) {
  // We are checking whether predecessor has gradients or not rather than
  // its an input ot not because,this way will be helpful to calculate
  // gradients for input in getInputGradientsSingle.
  if (!_predecessor->getOutputVector(vec_index).gradients) {
    _layer->backpropagateInputLayer(_predecessor->getOutputVector(vec_index),
                                    this->getOutputVectorImpl(vec_index));
  } else {
    _layer->backpropagate(_predecessor->getOutputVector(vec_index),
                          this->getOutputVectorImpl(vec_index));
  }
}

void FullyConnectedNode::updateParametersImpl(float learning_rate,
                                              uint32_t batch_cnt) {
  // TODO(Nicholas): Abstract away these constants
  _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
}

BoltVector& FullyConnectedNode::getOutputVectorImpl(uint32_t vec_index) {
  return (*_outputs)[vec_index];
}

std::vector<NodePtr> FullyConnectedNode::getPredecessorsImpl() const {
  return {_predecessor};
}

void FullyConnectedNode::summarizeImpl(std::stringstream& summary,
                                       bool detailed) const {
  summary << _predecessor->name() << " -> " << name() << " (FullyConnected): ";
  _layer->buildLayerSummary(summary, detailed);
}

Node::NodeState FullyConnectedNode::getState() const {
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
      "FullyConnectedNode is in an invalid internal state");
}

template <class Archive>
void FullyConnectedNode::serialize(Archive& archive) {
  archive(cereal::base_class<Node>(this), _layer, _config, _predecessor);
}

template void FullyConnectedNode::serialize(
    cereal::BinaryOutputArchive& archive);

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedNode)
