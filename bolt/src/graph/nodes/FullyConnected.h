#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerUtils.h>
#include <exceptions/src/Exceptions.h>
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

  void initializeParameters() final {
    if (!predecessorsSet()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call initializeParameters before setting predecessor of "
          "FullyConnectedNode.");
    }
    if (parametersInitialized()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call initializeParameters twice for FullyConnectedNode.");
    }

    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

  std::shared_ptr<FullyConnectedNode> addPredecessor(NodePtr node) {
    if (predecessorsSet()) {
      throw exceptions::NodeStateMachineError(
          "FullyConnectedNode expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);

    return shared_from_this();
  }

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    assert(preparedForBatchProcessing());

    _layer->forward(_predecessor->getOutputVector(batch_index),
                    this->getOutputVector(batch_index), labels);
  }

  void backpropagate(uint32_t batch_index) final {
    assert(preparedForBatchProcessing());

    // TODO(Nicholas, Josh): Change to avoid having this check
    if (_predecessor->isInputNode()) {
      _layer->backpropagateInputLayer(
          _predecessor->getOutputVector(batch_index),
          this->getOutputVector(batch_index));
    } else {
      _layer->backpropagate(_predecessor->getOutputVector(batch_index),
                            this->getOutputVector(batch_index));
    }
  }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    assert(preparedForBatchProcessing());

    // TODO(Nicholas): Abstract away these constants
    _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
  }

  BoltVector& getOutputVector(uint32_t batch_index) final {
    assert(preparedForBatchProcessing());

    return (*_outputs)[batch_index];
  }

  uint32_t outputDim() const final { return _config.dim; }

  uint32_t numNonzerosInOutput() const final {
    if (!preparedForBatchProcessing()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call numNonzerosInOutput before prepareForBatchProcessing in "
          "FullyConnectedNode.");
    }

    return (*_outputs)[0].len;
  }

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) final {
    if (!parametersInitialized()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call prepareForBatchProcessing before initializeParameters "
          "in FullyConnectedNode.");
    }

    if (preparedForBatchProcessing()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call prepareForBatchProcessing twice in a row in "
          "FullyConnectedNode");
    }

    // TODO(Nicholas): rename createBatchState
    _outputs =
        _layer->createBatchState(batch_size, /* use_sparsity=*/use_sparsity);
  }

  void cleanupAfterBatchProcessing() final {
    if (!preparedForBatchProcessing()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call cleanupAfterBatchProcessing before "
          "prepareForBatchProcessing in FullyConnectedNode.");
    }

    _outputs = std::nullopt;
  }

  std::vector<NodePtr> getPredecessors() const final {
    if (!predecessorsSet()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call getPredecessors before "
          "setting predecessors in FullyConnectedNode.");
    }

    return {_predecessor};
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    if (!parametersInitialized()) {
      throw exceptions::NodeStateMachineError(
          "Cannot call getInternalFullyConnectedLayers before "
          "initializeParameters in FullyConnectedNode.");
    }

    return {_layer};
  }

  bool isInputNode() const final { return false; }

  ActivationFunction getActivationFunction() const { return _config.act_func; }

 private:
  bool predecessorsSet() const { return _predecessor != nullptr; }

  bool parametersInitialized() const { return _layer != nullptr; }

  bool preparedForBatchProcessing() const { return _outputs.has_value(); }

  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  std::optional<BoltBatch> _outputs;

  NodePtr _predecessor;
};

}  // namespace thirdai::bolt