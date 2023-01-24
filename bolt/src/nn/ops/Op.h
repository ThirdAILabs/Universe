#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <memory>

namespace thirdai::bolt::nn::autograd {

class Computation;

using ComputationPtr = std::shared_ptr<Computation>;
using ComputationList = std::vector<ComputationPtr>;

}  // namespace thirdai::bolt::nn::autograd

namespace thirdai::bolt::nn::ops {

/**
 * Represents a operation in a model which takes in one or more inputs and
 * produces an output. The tensors which store the values for the inputs and
 * output should not be stored or referenced by the op and will be passed into
 * the op for forward and backpropagate. This is so the op is not fixed to a
 * particular location in a model and can be reused. For instance if you want to
 * use a FullyConnected op in a simple RNN, then the op itself is the same, but
 * the inputs and output it is called with changes. The op should store and
 * maintain any parameters it uses, for example a weight matrix or embedding
 * table, and it can store information about expected dimensions that are
 * invariant across different inputs it may be called with.
 */
class Op {
 public:
  explicit Op(std::string name) : _name(std::move(name)) {}

  /**
   * Computes the forward computation of the op. This should use the inputs in
   * the given set of input tensors and store the result in the given output
   * tensor. The parameter index_in_batch indicates which sample of the batch
   * the computation is for. This allows the model to parallelize the entire
   * forward and/or backward pass through the graph across the batch. Thus, this
   * function should be thread safe to being called with different values for
   * index_in_batch at the same time.
   */
  virtual void forward(const autograd::ComputationList& inputs,
                       tensor::TensorPtr& output, uint32_t index_in_batch,
                       bool training) = 0;

  /**
   * Computes the gradients of any parameters in the op and with respect to the
   * output tensor, and backpropagates gradients from the output tensor to the
   * inputs. The op should increment the gradients of its inputs so that it does
   * not overwrite existing gradients other ops may compute for the same
   * tensor. The model ensures that backpropagate will will not be called on a
   * given op and its output until all the ops that backpropagate gradients to
   * that output have done so. The parameter index_in_batch indicates which
   * sample of the batch the computation is for. This allows the model to
   * parallelize the entire forward and/or backward pass through the graph
   * across the batch. Thus, this function should be thread safe to being called
   * with different values for index_in_batch at the same time (though benign
   * race conditions to e.g. weight array are sometimes okay for performance).
   */
  virtual void backpropagate(autograd::ComputationList& inputs,
                             tensor::TensorPtr& output,
                             uint32_t index_in_batch) = 0;

  /**
   * Performs a parameter update on any parameters in the op. The parameter
   * train steps represents how many train steps have been completed so far in
   * the model. This is for logging and also optimizers like Adam which requires
   * this for bias correction.
   */
  virtual void updateParameters(float learning_rate, uint32_t train_steps) = 0;

  virtual uint32_t dim() const = 0;

  /**
   * Returns the number of nonzeros in the ops output for a given list of
   * inputs that will be passed to the op's forward computation. The
   * use_sparsity argument indicates whether sparsity will be used in the
   * forward computation. There are several reasons why the number of nonzeros
   * may be dependent on these arguments. If the op is a FullyConnected op using
   * a sparse layer the number of nonzeros in the output will be dependent on
   * whether or not sparsity is being used. If the op is a Concatenate op with
   * sparse inputs, then if sparsity is being used the number of nonzeros in the
   * output will depend on the number of nonzeros in the inputs.
   */
  virtual std::optional<uint32_t> nonzeros(
      const autograd::ComputationList& inputs, bool use_sparsity) const = 0;

  /**
   * Disables sparse parameter updates for updateParameters in the op. This is
   * used for distributed and also can be beneficial in cases where most of the
   * parameters are being updated and dense updates are faster.
   */
  virtual void disableSparseParameterUpdates() = 0;

  /**
   * Appends a line to the summary to describe the op when applied to the given
   * inputs and yielding the given output.
   */
  virtual void summary(std::ostream& summary,
                       const autograd::ComputationList& inputs,
                       const autograd::Computation* output) const = 0;

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