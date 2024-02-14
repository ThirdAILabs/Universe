#pragma once
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt {

class Computation;
using ComputationPtr = std::shared_ptr<Computation>;
using ComputationList = std::vector<ComputationPtr>;

/**
 * A Computation represents a node in the overarching computation graph for the
 * Bolt model. It stores a single Op that defines how the Computation processes
 * its input, a Tensor representing the output of the Computation, and a list of
 * Computations that precede it and make up the input. It is used to store the
 * computation graph and determine the correct ordering of the ops for autograd.
 */
class Computation {
 public:
  Computation(OpPtr op, ComputationList inputs);

  static ComputationPtr make(OpPtr op, ComputationList inputs);

  /**
   * Returns the op which operates on the inputs and output of the
   * computation.
   */
  OpPtr op() const;

  /**
   * Returns the inputs to the computation.
   */
  const ComputationList& inputs() const;

  /**
   * Returns the output of the computation.
   */
  TensorPtr& tensor();

  /**
   * Computes the activations of the neurons in the output of the computation
   * from its inputs using its source op. Calls the forward method of the source
   * op. The parameter index_in_batch indicates which sample of the batch
   * the computation should process. This allows the model to parallelize the
   * entire forward and/or backward pass through the graph across the batch.
   */
  void forward(uint32_t index_in_batch, bool training);

  /**
   * Backpropagates the gradients of the outputs of the computation to its
   * inputs using the source op. Calls the backpropagate method of the source
   * op. The parameter index_in_batch indicates which sample of the batch the
   * computation should process. This allows the model to parallelize the entire
   * forward and/or backward pass through the graph across the batch.
   */
  void backpropagate(uint32_t index_in_batch);

  /**
   * Returns the output dimension of the computation.
   */
  uint32_t dim() const;

  /**
   * Returns the number of nonzeros the output tensor will contain depending on
   * whether or not sparsity is being used and the inputs. Calls the nonzeros
   * method of the source op. Returns std::nullopt if the number of nonzeros is
   * not fixed, for instance in a sparse input.
   */
  std::optional<uint32_t> nonzeros(bool use_sparsity) const;

  /**
   * Reallocates the output tensor to reflect either a change in the batch size
   * the model is processing, a change in whether sparsity is being used for the
   * computations, or a change in the sparsity of some op in the model. This
   * method obtains its number of nonzeros from its source op by passing in the
   * inputs and whether sparsity is enabled.
   */
  void allocate(uint32_t batch_size, bool use_sparsity);

  /**
   * Adds an additional input to the end of the input list which will be passed
   * into the computation's op during forward and backward. One possible use
   * case is to add the labels as an extra input to the FullyConnected op when
   * the FullyConnected op yields an output and we want the labels to always be
   * in the set of active neurons.
   */
  void addInput(ComputationPtr input);

  /**
   * Sets the tensor representing the output of the computation. This is used to
   * inject data into the computation graph, for example the inputs and labels
   * during training.
   */
  void setTensor(TensorPtr tensor);

  /**
   * Outputs a summary of the computation to the given output stream.
   */
  void summary(std::ostream& summary);

  /**
   * Returns the name assigned to the computation.
   */
  const std::string& name() const;

  void setName(const std::string& name);

  std::vector<std::string> inputNames() const;

 private:
  OpPtr _op;

  ComputationList _inputs;

  TensorPtr _output;

  std::string _name;

  Computation() {}

  friend class cereal::access;
  // Because inputs are also computations clang-tidy things this is an infinite
  // recursive loop because eventually the serialize function for the input
  // computations are called within the serialize function for this computation.
  template <class Archive>
  void serialize(Archive& archive);  // NOLINT
};

}  // namespace thirdai::bolt