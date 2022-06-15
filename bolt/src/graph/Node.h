#pragma once

#include "Graph.h"
#include <queue>
#include <stdexcept>

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
  virtual bool hasSparseOutput() const = 0;

  // Returns the sparse output size of the node. If the node is dense then this
  // should be equivalent to outputDim().
  virtual uint32_t sparseOutputDim() const = 0;

  // Initializes any state that the node must store for computations that is not
  // part of the nodes parameters. For instance this could be the
  // activations/gradients for a batch, or some other internal state that must
  // be allocated after the batch size is known.
  virtual void initializeState(uint32_t batch_size, bool use_sparsity) = 0;

  // Enqueues any predecessors of the node. This is used to traverse the graph
  // during compilation.
  virtual void enqueuePredecessors(std::queue<NodePtr>& nodes) = 0;

  // Add any sparse layers in the node to a list of sparse layers. A list of all
  // the sparse layers in the network use useful when rebuilding hash tables or
  // hash functions and when enabling sparse inference.
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

}  // namespace thirdai::bolt