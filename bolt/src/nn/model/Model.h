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
   * Updates the parameters of all ops.
   */
  void updateParameters(float learning_rate);

  /**
   * Returns the list of ops in the model in the order they will be executed
   * during the forward pass.
   */
  std::vector<ops::OpPtr> opComputationOrder() const;

  /**
   * Returns the list of computations in the model in the order they will be
   * computed during the forward pass.
   */
  autograd::ComputationList computationOrder() const;

  const autograd::ComputationList& outputs() const;

  /**
   * Sets the given labels as the current labels for the model. These are public
   * so they can be used by the trainer to set labels before computing metrics
   * during validation.
   * TODO(Nicholas) expose validate on batch to get around this.
   */
  uint32_t setLabels(const tensor::TensorList& label_batches);

  void setSingleLabel(const tensor::TensorPtr& labels);

  /**
   * Retrieves on op by name. Throws if not found.
   */
  ops::OpPtr getOp(const std::string& name) const;

  autograd::ComputationPtr getComputation(const std::string& name) const;

  /**
   * Returns the input computation that stores the labels for a given output
   * computation. Attempts to find an output computation with the given name
   * whose gradients are computed in a loss function which depends on no other
   * output computation. The reason for this is that if a loss function is
   * computed directly on the computation and a label vector we assume that the
   * label vector maps directly to the neurons in the output computation.
   * Returns nullptr if not such output computation is found.
   */
  autograd::ComputationPtr getLabelsForOutput(const std::string& output_name);

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
   * These methods perform checks to make sure that model is valid.
   */
  void checkNoOutputsHaveDependentOps() const;

  void checkAllOutputsAreUsedInLosses() const;

  /**
   * When a loss is applied to a single output computation coming from a fully
   * connected op this method connects that op with the corresponding labels in
   * the loss function so that the labels can be selected as active neurons when
   * the layer is sparse.
   */
  void matchOutputFullyConnectedLayersWithLabels();

  autograd::ComputationList _inputs;
  autograd::ComputationList _label_inputs;
  autograd::ComputationList _outputs;
  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _ops;
  autograd::ComputationList _computation_order;

  AllocationManager _allocation_manager;

  uint32_t _train_steps;
};

using ModelPtr = std::shared_ptr<Model>;

}  // namespace thirdai::bolt::nn::model