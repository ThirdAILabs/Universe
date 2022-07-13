#pragma once

#include "FullyConnected.h"
#include "TokenInput.h"
#include <bolt/src/graph/Node.h>
#include <memory>
#include <vector>

namespace thirdai::bolt {

class SwitchNode final : public Node,
                         public std::enable_shared_from_this<SwitchNode> {
 public:
  SwitchNode(uint32_t dim, const std::string& activation, uint32_t n_layers)
      : _layers_used(n_layers, false), _token_input(nullptr) {
    for (uint32_t i = 0; i < n_layers; i++) {
      _layers.push_back(std::make_shared<FullyConnectedNode>(dim, activation));
    }
  }

  SwitchNode(uint32_t dim, float sparsity, const std::string& activation,
             uint32_t n_layers)
      : _layers_used(n_layers, false), _token_input(nullptr) {
    for (uint32_t i = 0; i < n_layers; i++) {
      _layers.push_back(
          std::make_shared<FullyConnectedNode>(dim, sparsity, activation));
    }
  }

  uint32_t outputDim() const final { return _layers.at(0)->outputDim(); }

  bool isInputNode() const final { return false; }

  std::shared_ptr<SwitchNode> addPredecessors(NodePtr predecessor,  // NOLINT
                                              TokenInputPtr token_input) {
    for (auto& layer : _layers) {
      layer->addPredecessor(predecessor);
    }

    _token_input = std::move(token_input);

    return shared_from_this();
  }

 private:
  void compileImpl() final {
    for (auto& layer : _layers) {
      layer->compileImpl();
    }
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    std::vector<std::shared_ptr<FullyConnectedLayer>> fc_layers;
    for (const auto& layer : _layers) {
      fc_layers.push_back(layer->getInternalFullyConnectedLayers().at(0));
    }
    return fc_layers;
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    for (auto& layer : _layers) {
      layer->prepareForBatchProcessing(batch_size, use_sparsity);
    }
  }

  uint32_t numNonzerosInOutputImpl() const final {
    return _layers.at(0)->numNonzerosInOutput();
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    uint32_t active_layer = getActiveLayer(vec_index);
    _layers_used[active_layer] = true;
    _layers.at(active_layer)->forward(vec_index, labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    uint32_t active_layer = getActiveLayer(vec_index);
    _layers.at(active_layer)->backpropagate(vec_index);
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    for (uint32_t i = 0; i < _layers.size(); i++) {
      if (_layers_used[i]) {
        _layers[i]->updateParameters(learning_rate, batch_cnt);
        _layers_used[i] = false;
      }
    }
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    uint32_t active_layer = getActiveLayer(vec_index);
    return _layers.at(active_layer)->getOutputVector(vec_index);
  }

  void cleanupAfterBatchProcessingImpl() final {
    for (auto& layer : _layers) {
      layer->cleanupAfterBatchProcessing();
    }
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    summary << name() << " (SwitchLayer) : ";
    _layers.at(0)->getInternalFullyConnectedLayers().at(0)->buildLayerSummary(
        summary, detailed);
  }

  std::string type() const final { return "switch"; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    auto predecessors = _layers[0]->getPredecessors();
    predecessors.push_back(_token_input);
    return predecessors;
  }

  NodeState getState() const final { return _layers.at(0)->getState(); }

  uint32_t getActiveLayer(uint32_t vec_index) {
    return _token_input->getTokens(vec_index).at(0);
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _layers, _layers_used,
            _token_input);
  }

  std::vector<std::shared_ptr<FullyConnectedNode>> _layers;
  std::vector<bool> _layers_used;
  TokenInputPtr _token_input;
};

}  // namespace thirdai::bolt