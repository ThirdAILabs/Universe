#pragma once

#include <cereal/access.hpp>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt_vector/src/BoltVector.h>
#include <exceptions/src/Exceptions.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

class Node;
// Node objects should always be initialized as shared pointers and not raw
// Nodes, since otherwise shared_from_this() might throw an error (we need
// shared_from_this for a clean functional style python api)
using NodePtr = std::shared_ptr<Node>;

// This class keeps track of the node types that have been traversed so that
// each node can get a unique name.
class LayerNameManager {
 public:
  /*
   * Example usage:
   * registerNodeAndGetName("concat") -> concat_1
   * registerNodeAndGetName("input") -> input_1
   * registerNodeAndGetName("concat") -> concat_2
   * registerNodeAndGetName("fc") -> fc_1
   * registerNodeAndGetName("input") -> input_2
   */
  std::string registerNodeAndGetName(const std::string& node_type) {
    type_to_count[node_type] += 1;
    std::string name =
        node_type + "_" + std::to_string(type_to_count[node_type]);
    return name;
  }

 private:
  std::unordered_map<std::string, uint32_t> type_to_count;
};

/*
  This class represents the interface used for nodes in a graph. It acts as a
  state machine for the node implementation to make sure that it is used
  correctly in the graph, and that the key parts of its initialization are
  called in the correct order. It has the following states:
    1. Constructed - the node object is created.
    2. Predecessors set - the predecessor nodes (if any) of the node are set.
    3. Compiled - the nodes name is set, and any other parameters that do not
       change during the node's lifetime are set (e.g. weight matrices,
       embedding tables, hash tables, etc.).
    4. Prepared for batch processing - in this state any temporary data
       structures for maintaining the state of the node are created. Most
       commonly this will be allocating arrays for the activations and
       gradients. In this state methods like forward, backward, etc. can be
       called.

  This node parent class will perform checks to ensure that the correct methods
  are called from the correct states and that the states are traversed in the
  correct order.
*/
class Node {
 public:
  /*
   * Compiles a single Node, including initializing any parameters and setting
   * the Node's name. The Node should use the passed in LayerNameManager to get
   * the name for its Node type. This moves the Node from state 2 to state 3.
   */
  void compile(LayerNameManager& name_manager);

  /*
   * Computes the forward pass for the node. The node will access its inputs
   * through the getOutput() method on is predecessor(s). The labels are an
   * optional argument that will only be specified for the output layer in order
   * for sparse layers to sample the labels as active neurons during training.
   * This forward pass must fill the output vector specified by
   * getOutputVector(vec_index), including setting the gradients equal to 0
   * (so they can be += to correctly in backpropogate in succesor nodes).
   */
  inline void forward(uint32_t vec_index, const BoltVector* labels) {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    forwardImpl(vec_index, labels);
  }

  // Computes the backwards pass through the node.
  inline void backpropagate(uint32_t vec_index) {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    backpropagateImpl(vec_index);
  }

  // Updates any trainable parameters
  inline void updateParameters(float learning_rate, uint32_t batch_cnt) {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    updateParametersImpl(learning_rate, batch_cnt);
  }

  // Returns the ith output of the node.
  inline BoltVector& getOutputVector(uint32_t vec_index) {
    assert(getState() == NodeState::PreparedForBatchProcessing);
    return getOutputVectorImpl(vec_index);
  }

  // Returns the output dimension of the node. This is used for subsequent nodes
  // during compilation.
  virtual uint32_t outputDim() const = 0;

  /*
   * Returns the number of nonzeros this node will have in its output. If the
   * node is dense then this will be equal to outputDim(). If the node is sparse
   * and the current network is prepared for sparse, this will return the sparse
   * dimension, the number of neurons this node will select during training or
   * inference. If the node is sparse and the network is prepared for dense,
   * this will be equal to outputDim(). If this quantity is unknowable, this
   * will throw an error. Currently, it is only unknowable for the Input node,
   * so it is the responsibility of the caller to call isInputNode() first.
   */
  uint32_t numNonzerosInOutput() const;

  /*
    Initializes any state that the node must store for computations that is not
    part of the nodes parameters. For instance this could be the
    activations/gradients for a batch, or some other internal state that must
    be allocated after the batch size is known. This needs to be called before
    training or inference with a new set of batches, since the batch size might
    change in different calls to predict or train.

    This moves the node from state 3 to state 4.
  */
  void prepareForBatchProcessing(uint32_t batch_size, bool use_sparsity);

  /*
   * Do any cleanup to bring the Node into the same state it was in before
   * prepareForBatchProcessing was called. This moves the node from state 4 to
   * state 3.
   */
  void cleanupAfterBatchProcessing();

  // Returns any predecessors of the node. This is used to traverse the graph
  // during compilation.
  std::vector<NodePtr> getPredecessors() const;

  /*
    Returns any fully connected layer objects used by the node. This list is
    needed by the graph object to perform sparsity specific operations on the
    entire network, like rebuilding all hash tables or reinitializing hash
    functions after a certain number of batches.
  */
  std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayers();

  // Returns true if the node is an input node.
  virtual bool isInputNode() const = 0;

  virtual void initOptimizer() = 0;

  // Prints out a single line summary in the format
  // (pred_names) -> node_name (NodeType): parameter_1=1, parameter_2=0 ...
  void summarize(std::stringstream& summary, bool detailed) const;

  // Returns the name of this node (only valid after the node has been
  // compiled).
  const std::string& name() const;

  /**
   * This function prevents nodes from using sparse optimizations during
   * parameter updates. This is to make updateParameters work during distributed
   * training or disable the optimization in the few cases where they are not
   * beneficial. By default it is a NOOP but nodes can override if they have
   * parameters that they apply sparse updates to.
   */
  virtual void disableSparseParameterUpdates() {}

  virtual ~Node() = default;

  /**
   * This function indicate whether this particular node needs to share its
   * gradients in distributed setting. If it is true, then make sure to
   * implement the logic to flattening the gradients in
   * GradientReference(bolt/python_bindings/BoltNNPython.h).
   */
  virtual bool hasParameters() = 0;

  // Return a short all lowercase string representing the type of this node for
  // use in printing the graph, e.g. concat, fc, input
  virtual std::string type() const = 0;

 protected:
  virtual void compileImpl() = 0;

  virtual std::vector<std::shared_ptr<FullyConnectedLayer>>
  getInternalFullyConnectedLayersImpl() const = 0;

  virtual void prepareForBatchProcessingImpl(uint32_t batch_size,
                                             bool use_sparsity) = 0;

  virtual uint32_t numNonzerosInOutputImpl() const = 0;

  virtual void forwardImpl(uint32_t vec_index, const BoltVector* labels) = 0;

  virtual void backpropagateImpl(uint32_t vec_index) = 0;

  virtual void updateParametersImpl(float learning_rate,
                                    uint32_t batch_cnt) = 0;

  virtual BoltVector& getOutputVectorImpl(uint32_t vec_index) = 0;

  virtual void cleanupAfterBatchProcessingImpl() = 0;

  virtual std::vector<NodePtr> getPredecessorsImpl() const = 0;

  virtual void summarizeImpl(std::stringstream& summary,
                             bool detailed) const = 0;

  enum NodeState {
    Constructed,
    PredecessorsSet,
    Compiled,
    PreparedForBatchProcessing
  };

  virtual NodeState getState() const = 0;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::optional<std::string> _name;
};

}  // namespace thirdai::bolt
