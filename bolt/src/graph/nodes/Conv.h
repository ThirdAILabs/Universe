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
  void compileImpl() final {
    assert(_config.has_value());
    _layer =
        std::make_shared<ConvLayer>(_config.value(), _predecessor->outputDim());
    _config = std::nullopt;
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size, bool use_sparsity) {
    _outputs =
        _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
  }

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutputVector(vec_index),
                    this->getOutputVectorImpl(vec_index), labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    if (_predecessor->isInputNode()) {
      _layer->backpropagateInputLayer(_predecessor->getOutputVector(vec_index),
                                      this->getOutputVectorImpl(vec_index));
    } else {
      _layer->backpropagate(_predecessor->getOutputVector(vec_index),
                            this->getOutputVectorImpl(vec_index));
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    // TODO(Nicholas): Abstract away these constants
    _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
  }

  BoltVector& getOutputVectorImpl(uint32_t vec_index) final {
    return (*_outputs)[vec_index];
  }

  void cleanupAfterBatchProcessingImpl() final { _outputs = std::nullopt; }

  std::vector<NodePtr> getPredecessorsImpl() const final {
    return {_predecessor};
  }

  void summarizeImpl(std::stringstream& summary, bool detailed) const final {
    summary << _predecessor->name() << " -> " << name() << " (Conv): ";
    _layer->buildLayerSummary(summary, detailed);
  }

  std::string type() const final { return "conv"; }

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
        "ConvNode is in an invalid internal state");
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

using ConvNodePtr = std::shared_ptr<ConvNode>;

}  // namespace thirdai::bolt