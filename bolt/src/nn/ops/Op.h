#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <memory>

namespace thirdai::bolt::nn::ops {

/**
 * Represents a operation in a model which takes in one or more inputs and
 * produces one or more outputs. The op is responsible for managing its own
 * parameters, for example the weight matrix in an fully connected layer.
 */
class Op {
 public:
  explicit Op(std::string name) : _name(std::move(name)) {}

  /**
   * Computes the forward computation of the op. This should use the vectors in
   * the op's input tensors as input and store the resulting activations in the
   * op's output tensor(s). The parameter index_in_batch indicates which sample
   * of the batch the computation is for. This allows the model to parallelize
   * the entire forward and backward pass through the graph across the batch.
   */
  virtual void forward(uint32_t index_in_batch) = 0;

  /**
   * Computes the gradients of the parameters in the op and the op's input with
   * respect to the gradient of the op's output tensor(s). The op should
   * increment the gradients of its inputs in case other ops compute gradients
   * for the same tensors. The parameter index_in_batch indicates which sample
   * of the batch the computation is for. This allows the model to parallelize
   * the entire forward and backward pass through the graph across the batch.
   */
  virtual void backpropagate(uint32_t index_in_batch) = 0;

  /**
   * Performs a parameter update on any parameters in the op. The parameter
   * train steps represents how many train steps have been completed so far in
   * the model. This is for logging and also optimizers like Adam which requires
   * this for bias correction.
   */
  virtual void updateParameters(float learning_rate, uint32_t train_steps) = 0;

  /**
   * Disables sparse parameter updates for updateParameters in the op. This is
   * used for distributed and also can be beneficial in cases where most of the
   * parameters are being updated and dense updates are faster.
   */
  virtual void disableSparseParameterUpdates() = 0;

  /**
   * Returns the input tensor(s) of the op. The inputs here is stored as a raw
   * pointers instead of smart pointers to avoid cycles since the input tensors
   * will store smart pointers to the ops that use them in their dependent_ops
   * field. The graph only stores smart pointers in the forward direction in the
   * graph and raw pointers in the backward direction to avoid cycles.
   */
  virtual std::vector<tensor::Tensor*> inputs() const = 0;

  /**
   * Returns the output tensor(s) of the op.
   */
  virtual std::vector<tensor::ActivationTensorPtr> outputs() const = 0;

  /**
   * Indicates to the op that the sparsity of one of its inputs has changed.
   * This method is called on the dependent ops of any activation tensor
   * when its number of sparse nonzeros changes. This is because for certain
   * ops the number of nonzeros in its input may affect its output. For
   * example in a concatenation op if one of the inputs increases its
   * sparsity and hence its number of nonzeros then the number of nonzeros
   * in the concatenated output will change as well. For many ops this will
   * be a no-op.
   */
  virtual void notifyInputSparsityChange() = 0;

  /**
   * Returns a summary of the op.
   */
  virtual void summary(std::ostream& summary) const = 0;

  /**
   * Returns the name of the op. All of the ops in a model must have a unique
   * name.
   */
  const std::string& name() const { return _name; }

 private:
  std::string _name;
};

using OpPtr = std::shared_ptr<Op>;

}  // namespace thirdai::bolt::nn::ops