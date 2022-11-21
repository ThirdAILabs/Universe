#pragma once

#include <bolt/src/graph/Node.h>

namespace thirdai::bolt {

class ConvNode final : public Node,
                       public std::enable_shared_from_this<ConvNode> {
 private:
  ConvNode(uint64_t num_filters, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size)
      : _layer(nullptr),
        _config(
            ConvLayerConfig(num_filters, activation, std::move(kernel_size))),
        _predecessor(nullptr) {}

  ConvNode(uint64_t num_filters, float sparsity, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size)
      : _layer(nullptr),
        _config(ConvLayerConfig(num_filters, sparsity, activation,
                                std::move(kernel_size))),
        _predecessor(nullptr) {}

  ConvNode(uint64_t num_filters, float sparsity, const std::string& activation,
           std::pair<uint32_t, uint32_t> kernel_size,
           SamplingConfigPtr sampling_config)
      : _layer(nullptr),
        _config(ConvLayerConfig(num_filters, sparsity, activation,
                                std::move(kernel_size),
                                std::move(sampling_config))),
        _predecessor(nullptr) {}

 public:
  static std::shared_ptr<ConvNode> makeDense(
      uint32_t dim, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size) {
    return std::shared_ptr<ConvNode>(
        new ConvNode(dim, activation, kernel_size));
  }

  static std::shared_ptr<ConvNode> makeAutotuned(
      uint32_t dim, float sparsity, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size) {
    return std::shared_ptr<ConvNode>(
        new ConvNode(dim, sparsity, activation, kernel_size));
  }

  static std::shared_ptr<ConvNode> make(
      uint32_t dim, float sparsity, const std::string& activation,
      std::pair<uint32_t, uint32_t> kernel_size,
      SamplingConfigPtr sampling_config) {
    return std::shared_ptr<ConvNode>(new ConvNode(
        dim, sparsity, activation, kernel_size, std::move(sampling_config)));
  }

  std::shared_ptr<ConvNode> addPredecessor(NodePtr node) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "ConvNode expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);

    return shared_from_this();
  }

  uint32_t outputDim() const final {
    NodeState node_state = getState();
    if (node_state == NodeState::Constructed ||
        node_state == NodeState::PredecessorsSet) {
      return _config->getDim();
    }
    return _layer->getDim();
  }

  bool isInputNode() const final { return false; }

  void initOptimizer() final { _layer->initOptimizer(); }

 private:
  NodeState getState() const final {
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

  // Private constructor for cereal. Must create dummy config since no default
  // constructor exists for layer config.
  ConvNode() : _config(std::nullopt) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Node>(this), _layer, _config, _predecessor);
  }
  // One of _layer and _config will always be nullptr/nullopt while the
  // other will contain data
  std::shared_ptr<ConvLayer> _layer;
  std::optional<ConvLayerConfig> _config;
  std::optional<BoltBatch> _outputs;

  NodePtr _predecessor;
};

}  // namespace thirdai::bolt