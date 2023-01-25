#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/AllocationManager.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <vector>

namespace thirdai::bolt::nn::model {

class Model {
 public:
  Model(autograd::ComputationList inputs, autograd::ComputationList outputs,
        std::vector<loss::LossPtr> losses);

  static std::shared_ptr<Model> make(autograd::ComputationList inputs,
                                     autograd::ComputationList outputs,
                                     std::vector<loss::LossPtr> losses);

  /**
   * Computes the forward pass through the model for the given batch.
   * Activations are not cleared until the next call to forward or trainOnBatch.
   * Labels will not be selected by sparse fully connected layers which yield
   * outputs since labels are not provided. For this type of sampling
   * trainOnBatch must be used.
   */
  void forward(const tensor::TensorList& inputs, bool use_sparsity);

  void forward(const tensor::TensorPtr& inputs, bool use_sparsity);

  /**
   * Performs the foward and backward pass through the model for the given
   * training batch. There is no intermediate thread synchronization between ops
   * or forward/backpropagate. Does not perform parameter updates. Labels will
   * be selected by sparse fully connected layers which yield outputs.
   */
  void trainOnBatch(const tensor::TensorList& inputs,
                    const tensor::TensorList& labels);

  void trainOnBatch(const tensor::TensorPtr& inputs,
                    const tensor::TensorPtr& labels);

  /**
   * Performs the forward pass through the model on a given batch. Differs from
   * forward because the labels can be provided so that they are set and can
   * thus be used in metrics during validation.
   */
  void validateOnBatch(const tensor::TensorList& inputs,
                       const tensor::TensorList& labels, bool use_sparsity);

  void validateOnBatch(const tensor::TensorPtr& inputs,
                       const tensor::TensorPtr& labels, bool use_sparsity);

  /**
   * Updates the parameters in the model. Loops through all ops in the model
   * and calls updateParameters on each so that each op performs one
   * optimizer step on all of its trainable parameters.
   */
  void updateParameters(float learning_rate);

  /**
   * Returns the list of ops in the model in the order they will be executed
   * during the forward pass. Ops that are used multiple times will occur
   * multiple times in this list at each of the points in which they would be
   * executed.
   */
  std::vector<ops::OpPtr> opComputationOrder() const;

  /**
   * Returns the list of computations in the model in the order they will be
   * computed during the forward pass. Differs from opComputationOrder because
   * ops may be reused by all computations are unique.
   */
  autograd::ComputationList computationOrder() const;

  /**
   * Returns the outputs of the outputs of the model.
   */
  const autograd::ComputationList& outputs() const;

  /**
   * Retrieves on op by name. Throws if not found.
   */
  ops::OpPtr getOp(const std::string& name) const;

  autograd::ComputationPtr getComputation(const std::string& name) const;

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
  void forward(uint32_t input_batch_size, bool use_sparsity);

  /**
   * Helper method for trainOnBatch and trainOnSingleInputBatch. Handles all of
   * the logic after setting the inputs and labels.
   */
  void trainOnBatch(uint32_t input_batch_size, uint32_t label_batch_size);

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
  uint32_t setInputs(const tensor::TensorList& input_batches);

  void setSingleInput(const tensor::TensorPtr& input);

  /**
   * Sets the given labels as the current labels for the model.
   */
  uint32_t setLabels(const tensor::TensorList& label_batches);

  void setSingleLabel(const tensor::TensorPtr& labels);

  /**
   * Returns the computations in the model that are used in loss functions.
   */
  autograd::ComputationList outputsUsedInLossFunctions() const;

  /**
   * When a loss is applied to a single output computation coming from a fully
   * connected op this method connects that op with the corresponding labels in
   * the loss function so that the labels can be selected as active neurons when
   * the layer is sparse.
   */
  void matchOutputFullyConnectedLayersWithLabels();

  /**
   * These methods perform checks to make sure that model is valid.
   */
  void checkNoOutputsHaveDependentOps() const;

  void checkAllOutputsInComputationOrder() const;

  void checkNoOutputsUsedInMultipleLosses() const;

  autograd::ComputationList _inputs;
  autograd::ComputationList _outputs;
  autograd::ComputationList _labels;
  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _ops;
  autograd::ComputationList _computation_order;

  AllocationManager _allocation_manager;

  uint32_t _train_steps;
};

using ModelPtr = std::shared_ptr<Model>;

}  // namespace thirdai::bolt::nn::model