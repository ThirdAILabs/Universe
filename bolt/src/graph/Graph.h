#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Node;
using NodePtr = std::shared_ptr<Node>;

class Input;
using InputPtr = std::shared_ptr<Input>;

class GraphContext : public std::enable_shared_from_this<GraphContext> {
 public:
  // The graph is constructed with a list of input layers, the order of these
  // input layers is used to define how training/test inputs are mapped to the
  // specific layers. Using the output node the graph can be traversed backwards
  // to discover a reverse ordering in which to execute the layers.
  GraphContext(std::vector<InputPtr> inputs, NodePtr output)
      : _output(std::move(output)), _inputs(std::move(inputs)) {}

  // When the layers are initially defined the only have information about their
  // own dimensions, parameters etc. During compile the layers can use the
  // information from their predecessor(s) such as output dim do fully
  // initialize their parameters. Note that successors could be added to nodes
  // as well if that is needed for certain layers to initialize. Additionally in
  // this function the different layers can preform different checks to make
  // sure that the network is properly formatted. For instance if
  // CategoricalCrossEntropy loss is used, then it can verify that the output
  // layer has a softmax activation.
  void compile(std::shared_ptr<LossFunction> loss);

  // Computes the forward pass through the graph.
  void forward(uint32_t batch_index, BoltBatch& inputs, BoltBatch& labels);

  // Computes the backward pass through the graph.
  void backward(uint32_t batch_index, BoltBatch& inputs);

  // This is to allow for pointers to the graph to be passed into the nodes
  // themselves to access information about the context of the graph.
  std::shared_ptr<GraphContext> getPtr() { return shared_from_this(); }

 private:
  // List of nodes(layers) in the order in which they should be computed.
  std::vector<NodePtr> _nodes;

  // Output layer.
  NodePtr _output;

  // Input layers. When train is called, the ith input is fed into the ith input
  // layer.
  std::vector<InputPtr> _inputs;

  // List of the sparse layers in the graph. This is so that when we want to do
  // things like enable sparse inference, update hash tables, or update hash
  // functions.
  std::vector<std::shared_ptr<FullyConnectedLayer>> _sparse_layers;

  // The loss function the graph was compliled with.
  std::shared_ptr<LossFunction> _loss;
};

using GraphContextPtr = std::shared_ptr<GraphContext>;

}  // namespace thirdai::bolt