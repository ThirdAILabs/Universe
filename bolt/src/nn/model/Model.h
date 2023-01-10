#pragma once

#include "AllocationManager.h"
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
   * Labels will not be selected by sparse fully connected layers which yield
   * outputs since labels are not provided. For this type of sampling
   * trainOnBatch must be used.
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
   * synchronization. Does not perform parameter updates. Labels will be
   * selected by sparse fully connected layers which yield outputs.
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

  const std::vector<tensor::ActivationTensorPtr>& tensors() const;

  /**
   * Sets the given labels as the current labels for the model. These are public
   * so they can be used by the trainer to set labels before computing metrics
   * during validation.
   */
  uint32_t setLabels(const std::vector<BoltBatch>& label_batches);

  void setSingleLabel(const BoltBatch& labels);

  /**
   * Retrieves on op by name. Throws if not found.
   */
  ops::OpPtr getOp(const std::string& name) const;

  /**
   * Retrieves a tensor by name. Throws if not found.
   */
  tensor::ActivationTensorPtr getTensor(const std::string& name) const;

  /**
   * Returns the input tensor that stores the labels for a given output tensor.
   * Attempts to find an output tensor with the given name whose gradients are
   * computed in a loss function which depends on no other output tensor. The
   * reason for this is that if a loss function is compute directly on the
   * tensor and a label vector we assume that the label vector maps directly to
   * the neurons in the output tensor. Returns nullptr if not such output tensor
   * is found.
   */
  tensor::InputTensorPtr getLabelsForOutput(const std::string& output_name);

  /**
   * Returns a list of the output tensors in the model.
   */
  const std::vector<tensor::ActivationTensorPtr>& outputs() const;

  /**
   * Prints/returns a summary of the model. Throws if no op is found.
   */

  std::string summary(bool print = true) const;

  /**
   * Returns how many train steps the model has taken. Used for logging in
   * trainer.
   */
  uint32_t trainSteps() const;

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
  void forwardVector(uint32_t index_in_batch, bool training);

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
   * Traverses the graph and determines the order in which the ops should be
   * executed.
   */
  void createOpSchedule();

  /**
   * Gets the in degrees for each op, which is the number of tensors they take
   * as input.
   */
  std::unordered_map<tensor::ActivationTensorPtr, uint32_t> getOutDegrees()
      const;

  /**
   * These methods perform checks to make sure that model is valid.
   */
  void checkNoOutputsHaveDependentOps() const;

  void checkAllOutputsAreUsedInLosses() const;

  /**
   * When a loss is applied to a single ActivationTensor coming from a fully
   * connected op this method connects that op with the corresponding labels in
   * the loss function so that the labels can be selected as active neurons when
   * the layer is sparse.
   */
  void matchOutputFullyConnectedLayersWithLabels();

  std::vector<tensor::InputTensorPtr> _inputs;
  std::vector<tensor::InputTensorPtr> _label_inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;
  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _ops;
  std::vector<tensor::ActivationTensorPtr> _activation_tensors;

  AllocationManager _allocation_manager;

  uint32_t _train_steps;
};

using ModelPtr = std::shared_ptr<Model>;

}  // namespace thirdai::bolt::nn::model