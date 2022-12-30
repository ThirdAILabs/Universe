#pragma once

#include "ActivationsManager.h"
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::model {

class Model {
 public:
  Model(std::vector<tensor::InputTensorPtr> inputs,
        std::vector<tensor::ActivationTensorPtr> outputs,
        std::vector<loss::LossPtr> losses);

  static std::shared_ptr<Model> make(
      std::vector<tensor::InputTensorPtr> inputs,
      std::vector<tensor::ActivationTensorPtr> outputs,
      std::vector<loss::LossPtr> losses);

  /**
   * Computes the forward pass through the model for the given batch.
   * Activations are not cleared until the next call to forward or trainOnBatch.
   */
  void forward(const std::vector<BoltBatch>& inputs, bool use_sparsity);

  void forwardSingleInput(const BoltBatch& inputs, bool use_sparsity);

  /**
   * Computes the backward pass through the model with the given labels, using
   * the actiations that are currently stored in the graph from the last call to
   * forward. Assumes that forward(...) has been called already.
   */
  void backpropagate(const std::vector<BoltBatch>& labels);

  void backpropagateSingleInput(const BoltBatch& labels);

  /**
   * Performs the foward and backward pass through the model for the given
   * training batch. The benefit of calling this method over forward(...)
   * followed by backpropagate(...) is that there is no intermediate thread
   * synchronization.
   */
  void trainOnBatch(const std::vector<BoltBatch>& inputs,
                    const std::vector<BoltBatch>& labels);

  void trainOnBatchSingleInput(const BoltBatch& inputs,
                               const BoltBatch& labels);

  /**
   * Updates the parameters of all ops.
   */
  void updateParameters(float learning_rate);

  /**
   * Returns the list of ops in the model in the order they will be executed
   * during the forward pass.
   */
  const std::vector<ops::OpPtr>& ops() const;

  /**
   * Prints/returns a summary of the model.
   */

  std::string summary(bool print = true) const;

 private:
  /**
   * Helper function for forward and forwardSingleInput. Handles all of
   * the logic after setting the inputs and labels.
   */
  void forwardImpl(uint32_t input_batch_size, bool use_sparsity);

  /**
   * Helper function for backpropagate and backpropagateSingleInput. Handles all
   * of the logic after setting the inputs and labels.
   */
  void backpropagateImpl(uint32_t label_batch_size);

  /**
   * Helper method for trainOnBatch and trainOnSingleInputBatch. Handles all of
   * the logic after setting the inputs and labels.
   */
  void trainOnBatchImpl(uint32_t input_batch_size, uint32_t label_batch_size);

  /**
   * Computes the forward pass through the model for the given sample in the
   * batch. Assumes that setInputs(...) has already been called.
   */
  void forwardVector(uint32_t index_in_batch);

  /**
   * Computes the backward pass through the model for the given sample in the
   * batch. Assumes that setInputs(...) and setLabels(...) have already been
   * called.
   */
  void backpropagateVector(uint32_t index_in_batch);

  /**
   * Sets the given batch as the inputs to the model.
   */
  uint32_t setInputs(const std::vector<BoltBatch>& input_batches);

  void setSingleInput(const BoltBatch& inputs);

  /**
   * Sets the given labels as the current labels for the model.
   */
  uint32_t setLabels(const std::vector<BoltBatch>& label_batches);

  void setSingleLabel(const BoltBatch& labels);

  /**
   * Traverses the graph and determines the order in which the ops should be
   * executed.
   */
  void createOpSchedule();

  /**
   * Gets the in degrees for each op, which is the number of tensors they take
   * as input.
   */
  std::unordered_map<ops::OpPtr, uint32_t> getInDegrees() const;

  /**
   * These methods perform checks to make sure that model is valid.
   */
  void checkNoOutputsHaveDependentOps() const;

  void checkOnlyOutputsHaveNoDependentOps() const;

  void checkAllOutputsAreUsedInLosses() const;

  std::vector<tensor::InputTensorPtr> _inputs;
  std::vector<tensor::InputTensorPtr> _label_inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;

  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _op_schedule;
  ActivationsManager _activations;

  uint32_t _train_steps;
};

using ModelPtr = std::shared_ptr<Model>;

}  // namespace thirdai::bolt::nn::model