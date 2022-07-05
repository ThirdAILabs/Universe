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
      : _config(std::forward<Args>(args)...), _predecessor(nullptr) {}

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

 private:
  void compileImpl(LayerNameManager& name_manager) final {
    if (_predecessor == nullptr) {
      throw std::invalid_argument(
          "FullyConnected layer expected to have exactly one predecessor.");
    }
    auto name = name_manager.registerNodeAndGetName(/* node_type = */ "full");

    auto layer = std::make_shared<FullyConnectedLayer>(
        _config, _predecessor->outputDim());

    _compile_state = CompileState(/* name = */ name, /* layer = */ layer);
  }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const final {
    return {_compile_state->layer};
  }

  void prepareForBatchProcessingImpl(uint32_t batch_size,
                                     bool use_sparsity) final {
    // TODO(Nicholas): rename createBatchState
    _outputs = _compile_state->layer->createBatchState(
        batch_size, /* use_sparsity=*/use_sparsity);
  }

  uint32_t numNonzerosInOutputImpl() const final { return (*_outputs)[0].len; }

  void forwardImpl(uint32_t vec_index, const BoltVector* labels) final {
    _compile_state->layer->forward(_predecessor->getOutputVector(vec_index),
                                   this->getOutputVectorImpl(vec_index),
                                   labels);
  }

  void backpropagateImpl(uint32_t vec_index) final {
    // TODO(Nicholas, Josh): Change to avoid having this check
    if (_predecessor->isInputNode()) {
      _compile_state->layer->backpropagateInputLayer(
          _predecessor->getOutputVector(vec_index),
          this->getOutputVectorImpl(vec_index));
    } else {
      _compile_state->layer->backpropagate(
          _predecessor->getOutputVector(vec_index),
          this->getOutputVectorImpl(vec_index));
    }
  }

  void updateParametersImpl(float learning_rate, uint32_t batch_cnt) final {
    // TODO(Nicholas): Abstract away these constants
    _compile_state->layer->updateParameters(learning_rate, batch_cnt, BETA1,
                                            BETA2, EPS);
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
    _compile_state->layer->buildLayerSummary(summary, detailed);
  }

  const std::string& nameImpl() const final { return _compile_state->name; }

  NodeState getState() const final {
    if (_predecessor == nullptr && !_compile_state.has_value() &&
        !_outputs.has_value()) {
      return NodeState::Constructed;
    }
    if (_predecessor != nullptr && !_compile_state.has_value() &&
        !_outputs.has_value()) {
      return NodeState::PredecessorsSet;
    }
    if (_predecessor != nullptr && _compile_state.has_value() &&
        !_outputs.has_value()) {
      return NodeState::Compiled;
    }
    if (_predecessor != nullptr && _compile_state.has_value() &&
        _outputs.has_value()) {
      return NodeState::PreparedForBatchProcessing;
    }
    throw exceptions::NodeStateMachineError(
        "Node is in an invalid internal state");
  }

  struct CompileState {
    // We have this constructor so clang tidy can check variable names
    CompileState(std::string name, std::shared_ptr<FullyConnectedLayer> layer)
        : name(std::move(name)), layer(std::move(layer)) {}

    std::string name;
    std::shared_ptr<FullyConnectedLayer> layer;
  };

  FullyConnectedLayerConfig _config;
  std::optional<BoltBatch> _outputs;
  std::optional<CompileState> _compile_state;

  NodePtr _predecessor;
};

}  // namespace thirdai::bolt