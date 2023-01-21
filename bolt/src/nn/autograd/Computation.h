#pragma once
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt::nn::autograd {

class Computation;
using ComputationPtr = std::shared_ptr<Computation>;
using ComputationList = std::vector<ComputationPtr>;

/**
 * A computation represents a node in the computation graph. It stores in inputs
 * to and outputs of the computation, as well as the op that defines the
 * computation. It is used to store the computation graph and determine the
 * correct ordering of the ops for autograd.
 */
class Computation {
 public:
  Computation(ops::OpPtr op, ComputationList inputs);

  static ComputationPtr make(ops::OpPtr op, ComputationList inputs);

  /**
   * Returns the op which operates on the inputs and output of the
   * computation.
   */
  ops::OpPtr op() const;

  /**
   * Returns the inputs to the computation.
   */
  const ComputationList& inputs() const;

  /**
   * Returns the output of the computation.
   */
  tensor::TensorPtr& tensor();

  const tensor::TensorPtr& tensor() const;

  /**
   * Computes the activations of the neurons in the output of the computation
   * from its inputs using its source op. Calls the forward method of the source
   * op.
   */
  void forward(uint32_t index_in_batch, bool training);

  /**
   * Backpropagates the gradients of the outputs of the computation to its
   * inputs using the source op. Calls the backpropagate method of the source
   * op.
   */
  void backpropagate(uint32_t index_in_batch);

  /**
   * Returns the output dimension of the computation.
   */
  uint32_t dim() const;

  /**
   * Returns the number of nonzeros the output tensor will contain depending on
   * wether or not sparsity is being used and the inputs. Calls the nonzeros
   * method of the source op.
   */
  std::optional<uint32_t> nonzeros(bool use_sparsity) const;

  /**
   * Reallocates the number of vectors stored in the output tensor to reflect
   * either a change in the batch size the model is processing, a change in
   * whether sparsity is being used for the computations, or a change in the
   * sparsity of some op in the model. This method obtains its number of
   * nonzeros from its source op by passing in the inputs and wether sparsity is
   * enabled.
   */
  void allocate(uint32_t batch_size, bool use_sparsity);

  /**
   * Adds an additional input to the computation which will be passed into its
   * source op during forward and backward. This is only intended to be used to
   * add the labels as an extra input to the FullyConnected op when the
   * FullyConnected op yields an output and we want the labels to always be in
   * the set of active neurons.
   */
  void addInput(ComputationPtr input);

  /**
   * Outputs a summary of the computation to the given output stream.
   */
  void summary(std::ostream& summary);

  /**
   * Returns the name assigned to the computation.
   */
  const std::string& name() const;

 private:
  ops::OpPtr _op;

  ComputationList _inputs;

  tensor::TensorPtr _output;

  std::string _name;
};

}  // namespace thirdai::bolt::nn::autograd