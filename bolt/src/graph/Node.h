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

  // Computes the forward pass for the node. The node will access its inputs
  // through the getOutput() method on is predecessor(s). The labels are an
  // optional argument that will only be specified for the output layer in order
  // for sparse layers to sample the labels as active neurons during training.
  virtual void forward(uint32_t batch_index, const BoltVector* labels) = 0;

  // Computes the backwards pass through the node.
  virtual void backpropagate(uint32_t batch_index) = 0;

  // Returns the ith output of the node.
  virtual BoltVector& getOutput(uint32_t batch_index) = 0;

  // Returns the output dimension of the node. This is used for subsequent nodes
  // during compilation.
  virtual uint32_t outputDim() const = 0;

  // Returns if the output of the node is sparse.
  virtual bool outputSparse() const = 0;

  // Initializes any state that the node must store for computations that is not
  // part of the nodes parameters. For instance this could be the
  // activations/gradients for a batch, or some other internal state that must
  // be allocated after the batch size is known.
  virtual void initializeState(uint32_t batch_size, bool is_inference) = 0;

  // Add any sparse layers in the node to a list of sparse layers.
  virtual void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) = 0;

  virtual ~Node() = default;

 protected:
  // Compilation for the node.
  virtual void compile() = 0;

  // Pointer to the graph object itself in case the node needs to access
  // additional context.
  GraphContextPtr _graph;
};

// A node subclass that contains a fully connected layer.
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

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    _layer->forward(_predecessor->getOutput(batch_index), _outputs[batch_index],
                    labels);
  }

  void backpropagate(uint32_t batch_index) final {
    _layer->backpropagate(_predecessor->getOutput(batch_index),
                          _outputs[batch_index]);
  }

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t outputDim() const final { return _layer->getDim(); }

  bool outputSparse() const final {
    // Need to check sparsity of layer.
    return false;
  }

  void initializeState(uint32_t batch_size, bool is_inference) final {
    _outputs = _layer->createBatchState(batch_size, is_inference);
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    sparse_layers.push_back(_layer);
  }

 protected:
  void compile() final {
    if (_predecessor == nullptr) {
      throw std::invalid_argument("");
    }

    _layer = std::make_shared<FullyConnectedLayer>(_config,
                                                   _predecessor->outputDim());
  }

 private:
  std::shared_ptr<FullyConnectedLayer> _layer;
  FullyConnectedLayerConfig _config;
  BoltBatch _outputs;

  NodePtr _predecessor;
};

class Concatenation final : public Node {
 public:
  explicit Concatenation(std::vector<NodePtr> inputs)
      : _input_nodes(std::move(inputs)),
        _concatenated_dim(0),
        _sparse_output(false) {}

  // This may be a no-op, or we may need to map sparse indices to disjoint
  // ranges.
  void forward(uint32_t batch_index, const BoltVector* labels) final;

  // This may be  no-op or we may need to map disjoint ranges of sparse indices
  // to the dim of each sub-layer.
  void backpropagate(uint32_t batch_index) final;

  BoltVector& getOutput(uint32_t batch_index) final {
    return _outputs[batch_index];
  }

  uint32_t outputDim() const final { return _concatenated_dim; }

  bool outputSparse() const final { return _sparse_output; }

  void initializeState(uint32_t batch_size, bool is_inference) final {
    // How do we handle sparsity in concatenation layers?
    _outputs = BoltBatch(_concatenated_dim, batch_size, is_inference);
  }

  void addSparseLayers(
      std::vector<std::shared_ptr<FullyConnectedLayer>>& sparse_layers) final {
    for (auto& node : _input_nodes) {
      node->addSparseLayers(sparse_layers);
    }
  }

 protected:
  void compile() final {
    for (auto& node : _input_nodes) {
      node->compile(this->_graph);
      _concatenated_dim += node->outputDim();
      _sparse_output = _sparse_output || node->outputSparse();
    }

    // How to allocate state if there are sparse inputs?
  }

 private:
  std::vector<NodePtr> _input_nodes;
  uint32_t _concatenated_dim;
  BoltBatch _outputs;
  bool _sparse_output;
};

// A node subclass for input layers. The input batch will be stored in this
// layer so that subsequent layers can access the inputs through its getOutput()
// method. This makes the interface simplier by generalizing the forward pass so
// that other layers always just access the outputs of the previous layer rather
// than have to worry if they they need to access an input directly or access
// the outputs of a previous layer.
class Input final : public Node {
 public:
  explicit Input(uint32_t expected_input_dim)
      : _expected_input_dim(expected_input_dim) {}

  void compile() final {}

  void forward(uint32_t batch_index, const BoltVector* labels) final {
    (void)labels;
    (void)batch_index;
  }

  void backpropagate(uint32_t batch_index) final { (void)batch_index; }

  void setInputs(BoltBatch* inputs) { _input_batch = inputs; }

  BoltVector& getOutput(uint32_t batch_index) final {
    return (*_input_batch)[batch_index];
  }

  uint32_t expectedInputDim() const { return _expected_input_dim; }

  uint32_t outputDim() const final { return _expected_input_dim; }

  bool outputSparse() const final {
    // Need to check sparsity of input.
    return false;
  }

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