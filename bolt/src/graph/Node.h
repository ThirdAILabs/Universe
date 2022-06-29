#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <queue>
#include <stdexcept>

namespace thirdai::bolt {

class Node;

// Node objects should always be initialized as shared pointers and not raw
// Nodes, since otherwise shared_from_this() might throw an error (we need
// shared_from_this for a clean functional style python api)
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
  virtual BoltVector& getOutputVector(uint32_t vec_index) = 0;

  // Returns the output dimension of the node. This is used for subsequent nodes
  // during compilation.
  virtual uint32_t outputDim() const = 0;

  /*
   * Returns the number of nonzeros this node will have in its output. If the
   * node is dense then this will be equal to outputDim(). If the node is sparse
   * this will return the sparse dimension, the number of neurons this node will
   * select during training or inference. If this quantity is unknowable, this
   * will throw an error. Currently, it is only unknowable for the Input node,
   * so it is the responsibility of the caller to call isInputNode() first.
   */
  virtual uint32_t numNonzerosInOutput() const = 0;

  /*
    Initializes any state that the node must store for computations that is not
    part of the nodes parameters. For instance this could be the
    activations/gradients for a batch, or some other internal state that must
    be allocated after the batch size is known. This needs to be called before
    training or inference with a new set of batches, since the batch size might
    change in different calls to predict or train.
  */
  virtual void prepareForBatchProcessing(uint32_t batch_size,
                                         bool use_sparsity) = 0;

  // Returns any predecessors of the node. This is used to traverse the graph
  // during compilation.
  virtual std::vector<NodePtr> getPredecessors() const = 0;

  /*
    Returns any fully connected layer objects used by the node. This list is
    needed by the graph object to perform sparsity specific operations on the
    entire network, like rebuilding all hash tables or reinitializing hash
    functions after a certain number of batches.
  */
  virtual std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers() const = 0;

  // Returns true if the node is an input node.
  virtual bool isInputNode() const = 0;

  virtual ~Node() = default;
};

}  // namespace thirdai::bolt