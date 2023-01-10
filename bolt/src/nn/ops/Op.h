#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <memory>

namespace thirdai::bolt::nn::tensor {

class ActivationTensor;

}  // namespace thirdai::bolt::nn::tensor

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
  virtual void forward(const tensor::TensorList& inputs,
                       tensor::ActivationTensor* output,
                       uint32_t index_in_batch, bool training) = 0;

  /**
   * Computes the gradients of the parameters in the op and the op's input with
   * respect to the gradient of the op's output tensor(s). The op should
   * increment the gradients of its inputs in case other ops compute gradients
   * for the same tensors. The parameter index_in_batch indicates which sample
   * of the batch the computation is for. This allows the model to parallelize
   * the entire forward and backward pass through the graph across the batch.
   */
  virtual void backpropagate(tensor::TensorList& inputs,
                             tensor::ActivationTensor* output,
                             uint32_t index_in_batch) = 0;

  /**
   * Performs a parameter update on any parameters in the op. The parameter
   * train steps represents how many train steps have been completed so far in
   * the model. This is for logging and also optimizers like Adam which requires
   * this for bias correction.
   */
  virtual void updateParameters(float learning_rate, uint32_t train_steps) = 0;

  virtual uint32_t numNonzerosInOutput(const tensor::TensorList& inputs,
                                       bool use_sparsity) const = 0;

  /**
   * Disables sparse parameter updates for updateParameters in the op. This is
   * used for distributed and also can be beneficial in cases where most of the
   * parameters are being updated and dense updates are faster.
   */
  virtual void disableSparseParameterUpdates() = 0;

  /**
   * Returns a summary of the op.
   */
  virtual void summary(std::ostream& summary, const tensor::TensorList& inputs,
                       const tensor::ActivationTensor* output) const = 0;

  /**
   * Returns the name of the op. All of the ops in a model must have a unique
   * name.
   */
  const std::string& name() const { return _name; }

  virtual ~Op() = default;

 private:
  std::string _name;
};

using OpPtr = std::shared_ptr<Op>;

}  // namespace thirdai::bolt::nn::ops