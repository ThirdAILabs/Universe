#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/LayerUtils.h>
#include <memory>
#include <utility>

namespace thirdai::bolt {

class FullyConnectedLayerNode final
    : public Node,
      public std::enable_shared_from_this<FullyConnectedLayerNode> {
 public:
  // This pattern means that any valid constructor for a
  // FullyConnectedLayerConfig can be used to initialize the
  // FullyConnectedLayerNode, and that the args are directly forwarded to the
  // constructor for the config.
  template <typename... Args>
  explicit FullyConnectedLayerNode(Args&&... args)
      : _layer(nullptr),
        _config(std::forward<Args>(args)...),
        _predecessor(nullptr) {}

  void initializeParameters() final {
    if (_predecessor == nullptr) {
      throw std::invalid_argument(
          "FullyConnected layer expected to have exactly one predecessor.");
    }

    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

  std::shared_ptr<FullyConnectedLayerNode> addPredecessor(NodePtr node) {
    if (_predecessor != nullptr) {
      throw std::invalid_argument(
          "FullyConnected layer expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);

    return shared_from_this();
  }

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutput(batch_index),
                    this->getOutput(batch_index), labels);
  }

  void backpropagate(uint32_t batch_index) final {
    if (_predecessor->isInputNode()) {
      _layer->backpropagateInputLayer(_predecessor->getOutput(batch_index),
                                      this->getOutput(batch_index));
    } else {
      _layer->backpropagate(_predecessor->getOutput(batch_index),
                            this->getOutput(batch_index));
    }
  }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA2, EPS);
  }

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t outputDim() const final { return _config.dim; }

  bool hasSparseOutput() const final { return _config.sparsity < 1.0; }

  uint32_t numNonzerosInOutput() const final {
    return _config.sparsity * _config.dim;
  }

  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity) final {
    _outputs =
        _layer->createBatchState(batch_size, /* force_dense=*/!use_sparsity);
  }

  std::vector<NodePtr> getPredecessors() const final { return {_predecessor}; }

  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const final {
    return {_layer};
  }

  bool isInputNode() const final { return false; }

  ActivationFunction getActivationFunction() const { return _config.act_func; }

 private:
  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  BoltBatch _outputs;

  NodePtr _predecessor;
};

}  // namespace thirdai::bolt