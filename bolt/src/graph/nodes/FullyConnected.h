#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerUtils.h>
#include <exceptions/src/Exceptions.h>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace thirdai::bolt {

class FullyConnectedNode final
    : public Node,
      public std::enable_shared_from_this<FullyConnectedNode> {
 public:
  // This pattern means that any valid constructor for a
  // FullyConnectedLayerConfig can be used to initialize the
  // FullyConnectedLayerNode, and that the args are directly forwarded to the
  // constructor for the config.
  template <typename... Args>
  explicit FullyConnectedNode(Args&&... args)
      : _layer(nullptr),
        _config(std::forward<Args>(args)...),
        _predecessor(nullptr) {}

  std::shared_ptr<FullyConnectedNode> addPredecessor(NodePtr node) {
    if (getState() != NodeState::Constructed) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);

    return shared_from_this();
  }

  uint32_t outputDim() const final { return _config.dim; }

  bool isInputNode() const final { return false; }

  ActivationFunction getActivationFunction() const { return _config.act_func; }

  float getSparsity() { return _config.getSparsity(); }

 private:
  void compileImpl() final {
    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {_layer};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    // TODO(Nicholas): rename createBatchState
    _outputs =
        _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
  }

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutputVector(vec_index),
                    this->getOutputVectorImpl(vec_index), labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    // TODO(Nicholas, Josh): Change to avoid having this check
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
    summary << _predecessor->name() << " -> " << name()
            << " (FullyConnected): ";
    _layer->buildLayerSummary(summary, detailed);
  }

  std::string type() const final { return "fc"; }

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
        "Node is in an invalid internal state");
  }

  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  std::optional<BoltBatch> _outputs;

  NodePtr _predecessor;
};

}  // namespace thirdai::bolt