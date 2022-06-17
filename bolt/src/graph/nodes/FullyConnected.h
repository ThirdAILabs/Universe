#pragma once

#include <bolt/src/graph/Node.h>

namespace thirdai::bolt {

class FullyConnectedLayerNode final : public Node {
 public:
  explicit FullyConnectedLayerNode(FullyConnectedLayerConfig config)
      : _layer(nullptr), _config(std::move(config)), _predecessor(nullptr) {}

  void compile() final {
    if (_predecessor == nullptr) {
      throw std::invalid_argument(
          "FullyConnected layer expected to have exactly one predecessor.");
    }

    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

  void addPredecessor(NodePtr node) {
    if (_predecessor != nullptr) {
      throw std::invalid_argument(
          "FullyConnected layer expected to have exactly one predecessor, and "
          "addPredecessor cannot be called twice.");
    }
    _predecessor = std::move(node);
  }

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutput(batch_index), _outputs[batch_index],
                    labels);
  }

  void backpropagate(uint32_t batch_index) final {
    if (_predecessor->isInputNode()) {
      _layer->backpropagateInputLayer(_predecessor->getOutput(batch_index),
                                      _outputs[batch_index]);
    } else {
      _layer->backpropagate(_predecessor->getOutput(batch_index),
                            _outputs[batch_index]);
    }
  }

  void updateParameters(float learning_rate, uint32_t batch_cnt) final {
    _layer->updateParameters(learning_rate, batch_cnt, BETA1, BETA1, EPS);
  }

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t outputDim() const final { return _config.dim; }

  bool hasSparseOutput() const final { return _config.sparsity < 1.0; }

  uint32_t sparseOutputDim() const final {
    return _config.sparsity * _config.dim;
  }

  void initializeState(uint32_t batch_size, bool use_sparsity) final {
    _outputs = _layer->createBatchState(batch_size, use_sparsity);
  }

  void enqueuePredecessors(std::queue<NodePtr>& nodes) final {
    nodes.push(_predecessor);
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    sparse_layers.push_back(_layer);
  }

  bool isInputNode() const final { return false; }

 private:
  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  BoltBatch _outputs;

  NodePtr _predecessor;
};

};  // namespace thirdai::bolt