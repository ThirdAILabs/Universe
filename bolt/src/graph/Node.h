#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <queue>
#include <stdexcept>

namespace thirdai::bolt {

class Node;

using NodePtr = std::shared_ptr<Node>;

class Node {
 public:
  virtual void initializeParameters() = 0;

  // Computes the forward pass for the node. The node will access its inputs
  // through the getOutput() method on is predecessor(s). The labels are an
  // optional argument that will only be specified for the output layer in order
  // for sparse layers to sample the labels as active neurons during training.
  virtual void forward(uint32_t vec_index, const BoltVector* labels) = 0;

  // Computes the backwards pass through the node.
  virtual void backpropagate(uint32_t vec_index) = 0;

  // Updates any trainable parameters
  virtual void updateParameters(float learning_rate, uint32_t batch_cnt) = 0;

  // Returns the ith output of the node.
  virtual BoltVector& getOutput(uint32_t vec_index) = 0;

  // Returns the output dimension of the node. This is used for subsequent nodes
  // during compilation.
  virtual uint32_t outputDim() const = 0;

  // Returns if the output of the node is sparse.
  virtual bool hasSparseOutput() const = 0;

  // Returns the sparse output size of the node. If the node is dense then this
  // should be equivalent to outputDim().
  virtual uint32_t numNonzerosInOutput() const = 0;

  // Initializes any state that the node must store for computations that is not
  // part of the nodes parameters. For instance this could be the
  // activations/gradients for a batch, or some other internal state that must
  // be allocated after the batch size is known.
  virtual void prepareForBatchProcessing(uint32_t batch_size,
                                         bool use_sparsity) = 0;

  // Enqueues any predecessors of the node. This is used to traverse the graph
  // during compilation.
  virtual std::vector<NodePtr> getPredecessors() const = 0;

  // Add any sparse layers in the node to a list of sparse layers. A list of all
  // the sparse layers in the network use useful when rebuilding hash tables or
  // hash functions and when enabling sparse inference.
  virtual std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const = 0;

  virtual bool isInputNode() const = 0;

  virtual ~Node() = default;
};

}  // namespace thirdai::bolt