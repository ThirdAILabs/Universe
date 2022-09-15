#pragma once

#include "FullyConnected.h"
#include "Input.h"
#include <bolt/src/graph/Node.h>
#include <exceptions/src/Exceptions.h>
#include <memory>
#include <utility>
#include <vector>

namespace thirdai::bolt {

/**
 * This class is used in MLM experiments. This node stores N fully connected
 * layers, and takes in a regular bolt vector input as well as a token. For each
 * input there should be only a single token in the range [0,N) that indicates
 * which of the fully connected layers to use for the input. For instance in a
 * MLM model this was used where the number of layers was equal to the maximum
 * number of tokens in the sentence and for each input the layer at the index of
 * the masked token was used. The idea being that it would learn a slightly
 * different representation for each token in the sentence.
 */
class SwitchNode final : public Node,
                         public std::enable_shared_from_this<SwitchNode> {
 private:
  SwitchNode(uint32_t dim, const std::string& activation, uint32_t n_layers)
      : _layers_used(n_layers, false), _token_input(nullptr) {
    for (uint32_t i = 0; i < n_layers; i++) {
      _layers.push_back(FullyConnectedNode::makeDense(dim, activation));
    }
  }

  SwitchNode(uint32_t dim, float sparsity, const std::string& activation,
             uint32_t n_layers)
      : _layers_used(n_layers, false), _token_input(nullptr) {
    for (uint32_t i = 0; i < n_layers; i++) {
      _layers.push_back(
          FullyConnectedNode::makeAutotuned(dim, sparsity, activation));
    }
  }

  SwitchNode(uint32_t dim, float sparsity, const std::string& activation,
             const SamplingConfigPtr& sampling_config, uint32_t n_layers)
      : _layers_used(n_layers, false), _token_input(nullptr) {
    for (uint32_t i = 0; i < n_layers; i++) {
      _layers.push_back(
          FullyConnectedNode::make(dim, sparsity, activation, sampling_config));
    }
  }

 public:
  static std::shared_ptr<SwitchNode> makeDense(uint32_t dim,
                                               const std::string& activation,
                                               uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, activation, n_layers));
  }

  static std::shared_ptr<SwitchNode> makeAutotuned(
      uint32_t dim, float sparsity, const std::string& activation,
      uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, sparsity, activation, n_layers));
  }

  static std::shared_ptr<SwitchNode> make(
      uint32_t dim, float sparsity, const std::string& activation,
      const SamplingConfigPtr& sampling_config, uint32_t n_layers) {
    return std::shared_ptr<SwitchNode>(
        new SwitchNode(dim, sparsity, activation, sampling_config, n_layers));
  }

  uint32_t outputDim() const final {
    // All layers are constructed identically so we can use _layers[0] here.
    return _layers.at(0)->outputDim();
  }

  bool isInputNode() const final { return false; }

  void initOptimizer() final {
    for (auto& layer : _layers) {
      layer->initOptimizer();
    }
  }

  std::shared_ptr<SwitchNode> addPredecessors(NodePtr predecessor,  // NOLINT
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

 private:
  void compileImpl() final {
    for (auto& layer : _layers) {
      // We use compile impl because the state is checked when compile() is
      // called on the switch node and we don't want to name each individual
      // sub-layer.
      layer->compileImpl();
    }
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    std::vector<std::shared_ptr<FullyConnectedLayer>> fc_layers;
    for (const auto& layer : _layers) {
      // Each layer only has one internal FullyConnectedLayer.
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
    // All layers are constructed identically so we can use _layers[0] here.
    return _layers.at(0)->numNonzerosInOutput();
  }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    uint32_t active_layer = getActiveLayer(vec_index);
    _layers.at(active_layer)->forward(vec_index, labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    uint32_t active_layer = getActiveLayer(vec_index);
    _layers_used[active_layer] = true;
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
    summary << _layers.at(0)->getPredecessorsImpl().at(0)->name();
    summary << " -> " << name() << " (SwitchLayer): n_layers=";
    summary << _layers.size() << ", ";
    _layers.at(0)->getInternalFullyConnectedLayers().at(0)->buildLayerSummary(
        summary, detailed);
  }

  std::string type() const final { return "switch"; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    // All layers are constructed identically so we can use _layers[0] here.
    auto predecessors = _layers.at(0)->getPredecessors();
    predecessors.push_back(_token_input);
    return predecessors;
  }

  NodeState getState() const final {
    // All layers are constructed identically and all method are called on all
    // layers, so we can use _layers[0] here.
    return _layers.at(0)->getState();
  }

  uint32_t getActiveLayer(uint32_t vec_index) {
    // There will only be one token indicating which layer to use.
    assert(_token_input->getOutputVector(vec_index).len == 1);
    return _token_input->getOutputVector(vec_index).active_neurons[0];
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _layers, _layers_used,
            _token_input);
  }

  std::vector<std::shared_ptr<FullyConnectedNode>> _layers;
  std::vector<bool> _layers_used;
  InputPtr _token_input;
};

}  // namespace thirdai::bolt