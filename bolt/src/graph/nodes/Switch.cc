#include "Switch.h"

namespace thirdai::bolt {

SwitchNode::SwitchNode(uint32_t dim, const std::string& activation,
                       uint32_t n_layers)
    : _layers_used(n_layers, false), _token_input(nullptr) {
  for (uint32_t i = 0; i < n_layers; i++) {
    _layers.push_back(FullyConnectedNode::makeDense(dim, activation));
  }
}

SwitchNode::SwitchNode(uint32_t dim, float sparsity,
                       const std::string& activation, uint32_t n_layers)
    : _layers_used(n_layers, false), _token_input(nullptr) {
  for (uint32_t i = 0; i < n_layers; i++) {
    _layers.push_back(
        FullyConnectedNode::makeAutotuned(dim, sparsity, activation));
  }
}

SwitchNode::SwitchNode(uint32_t dim, float sparsity,
                       const std::string& activation,
                       const SamplingConfigPtr& sampling_config,
                       uint32_t n_layers)
    : _layers_used(n_layers, false), _token_input(nullptr) {
  for (uint32_t i = 0; i < n_layers; i++) {
    _layers.push_back(
        FullyConnectedNode::make(dim, sparsity, activation, sampling_config));
  }
}

void SwitchNode::initOptimizer() {
  for (auto& layer : _layers) {
    layer->initOptimizer();
  }
}

std::shared_ptr<SwitchNode> SwitchNode::addPredecessors(
    NodePtr predecessor,  // NOLINT
    InputPtr token_input) {
  if (token_input->outputDim() != _layers.size()) {
    throw exceptions::GraphCompilationFailure(
        "Switch requires an Input with dimension the same as the number of "
        "switch layers but received Input with dimension " +
        std::to_string(token_input->outputDim()) + ".");
  }

  auto num_nonzeros_range = token_input->numNonZerosRange();
  if (!num_nonzeros_range || num_nonzeros_range.value().first != 1 ||
      num_nonzeros_range.value().second != 1) {
    std::stringstream ss;
    ss << "Switch requires an Input with a single nonzero to indicate which "
          "layer to use, but received Input with between "
       << num_nonzeros_range.value().first << " and "
       << num_nonzeros_range.value().second << " nonzeros.";

    throw exceptions::GraphCompilationFailure(ss.str());
  }

  for (auto& layer : _layers) {
    layer->addPredecessor(predecessor);
  }

  _token_input = std::move(token_input);

  return shared_from_this();
}

void SwitchNode::disableSparseParameterUpdates() {
  for (const auto& fc_node : _layers) {
    fc_node->disableSparseParameterUpdates();
  }
}

void SwitchNode::compileImpl() {
  for (auto& layer : _layers) {
    // We use compile impl because the state is checked when compile() is
    // called on the switch node and we don't want to name each individual
    // sub-layer.
    layer->compileImpl();
  }
}

std::vector<std::shared_ptr<FullyConnectedLayer>>
SwitchNode::getInternalFullyConnectedLayersImpl() const {
  std::vector<std::shared_ptr<FullyConnectedLayer>> fc_layers;
  for (const auto& layer : _layers) {
    // Each layer only has one internal FullyConnectedLayer.
    fc_layers.push_back(layer->getInternalFullyConnectedLayers().at(0));
  }
  return fc_layers;
}

void SwitchNode::prepareForBatchProcessingImpl(uint32_t batch_size,
                                               bool use_sparsity) {
  for (auto& layer : _layers) {
    layer->prepareForBatchProcessing(batch_size, use_sparsity);
  }
}

void SwitchNode::forwardImpl(uint32_t vec_index, const BoltVector* labels) {
  uint32_t active_layer = getActiveLayer(vec_index);
  _layers.at(active_layer)->forward(vec_index, labels);
}

void SwitchNode::backpropagateImpl(uint32_t vec_index) {
  uint32_t active_layer = getActiveLayer(vec_index);
  _layers_used[active_layer] = true;
  _layers.at(active_layer)->backpropagate(vec_index);
}

void SwitchNode::updateParametersImpl(float learning_rate, uint32_t batch_cnt) {
  for (uint32_t i = 0; i < _layers.size(); i++) {
    if (_layers_used[i]) {
      _layers[i]->updateParameters(learning_rate, batch_cnt);
      _layers_used[i] = false;
    }
  }
}

BoltVector& SwitchNode::getOutputVectorImpl(uint32_t vec_index) {
  uint32_t active_layer = getActiveLayer(vec_index);
  return _layers.at(active_layer)->getOutputVector(vec_index);
}

void SwitchNode::cleanupAfterBatchProcessingImpl() {
  for (auto& layer : _layers) {
    layer->cleanupAfterBatchProcessing();
  }
}

void SwitchNode::summarizeImpl(std::stringstream& summary,
                               bool detailed) const {
  summary << _layers.at(0)->getPredecessorsImpl().at(0)->name();
  summary << " -> " << name() << " (SwitchLayer): n_layers=";
  summary << _layers.size() << ", ";
  _layers.at(0)->getInternalFullyConnectedLayers().at(0)->buildLayerSummary(
      summary, detailed);
}

std::vector<NodePtr> SwitchNode::getPredecessorsImpl() const {
  // All layers are constructed identically so we can use _layers[0] here.
  auto predecessors = _layers.at(0)->getPredecessors();
  predecessors.push_back(_token_input);
  return predecessors;
}

Node::NodeState SwitchNode::getState() const {
  // All layers are constructed identically and all method are called on all
  // layers, so we can use _layers[0] here.
  return _layers.at(0)->getState();
}

uint32_t SwitchNode::getActiveLayer(uint32_t vec_index) {
  // There will only be one token indicating which layer to use.
  assert(_token_input->getOutputVector(vec_index).len == 1);
  return _token_input->getOutputVector(vec_index).active_neurons[0];
}

}  // namespace thirdai::bolt
