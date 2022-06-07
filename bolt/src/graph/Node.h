#pragma once

#include "Graph.h"

namespace thirdai::bolt {

class Node {
 public:
  Node() : _graph(nullptr) {}

  void compile(GraphContextPtr graph) {
    _graph = std::move(graph);
    compile();
  }

  virtual void forward(uint32_t batch_index) = 0;

  virtual void backpropagate(uint32_t batch_index) = 0;

  virtual BoltVector& getOutput(uint32_t batch_index) = 0;

  virtual uint32_t inputDim() const = 0;

  virtual uint32_t outputDim() const = 0;

  virtual void initializeState(uint32_t batch_size, bool is_inference) = 0;

  virtual void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) = 0;

  virtual ~Node() = default;

 protected:
  virtual void compile() = 0;

  GraphContextPtr _graph;
};

class FullyConnectedLayerNode final : public Node {
 public:
  explicit FullyConnectedLayerNode(FullyConnectedLayerConfig& config)
      : _layer(nullptr), _config(config), _predecessor(nullptr) {}

  void addPredecessor(NodePtr node) {
    if (_predecessor != nullptr) {
      throw std::invalid_argument("");
    }
    _predecessor = std::move(node);
  }

  void compile() final {
    if (_predecessor == nullptr) {
      throw std::invalid_argument("");
    }

    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

  void forward(uint32_t batch_index) final {
    _layer->forward(_predecessor->getOutput(batch_index), _outputs[batch_index],
                    &_graph->getLabels(batch_index));
  }

  void backpropagate(uint32_t batch_index) final {
    _layer->backpropagate(_predecessor->getOutput(batch_index),
                          _outputs[batch_index]);
  }

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t inputDim() const final { return _layer->getInputDim(); }

  uint32_t outputDim() const final { return _layer->getDim(); }

  void initializeState(uint32_t batch_size, bool is_inference) final {
    _outputs = _layer->createBatchState(batch_size, is_inference);
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    sparse_layers.push_back(_layer);
  }

 private:
  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  BoltBatch _outputs;

  NodePtr _predecessor;
};

class Input final : public Node {
 public:
  explicit Input(uint32_t expected_input_dim)
      : _expected_input_dim(expected_input_dim) {}

  void compile() final {}

  void forward(uint32_t batch_index) final { (void)batch_index; }

  void backpropagate(uint32_t batch_index) final { (void)batch_index; }

  void setInputs(BoltBatch* inputs) { _input_batch = inputs; }

  BoltVector& getOutput(uint32_t batch_index) final {
    return (*_input_batch)[batch_index];
  }

  uint32_t inputDim() const final { return _expected_input_dim; }

  uint32_t outputDim() const final { return _expected_input_dim; }

  void initializeState(uint32_t batch_size, bool is_inference) final {
    (void)batch_size;
    (void)is_inference;
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    (void)sparse_layers;
  }

 private:
  BoltBatch* _input_batch;
  uint32_t _expected_input_dim;
};

}  // namespace thirdai::bolt