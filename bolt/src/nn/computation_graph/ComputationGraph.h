#pragma once

#include "ActivationsManager.h"
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::computation_graph {

class ComputationGraph {
 public:
  ComputationGraph(std::vector<tensor::InputTensorPtr> inputs,
                   std::vector<tensor::ActivationTensorPtr> outputs,
                   std::vector<loss::LossPtr> losses);

  /**
   * Computes the forward pass through the computation graph for the given
   * batch. Activations are not cleared until the next call to forward or
   * trainOnBatch.
   */
  void forward(const std::vector<BoltBatch>& inputs, bool use_sparsity);

  /**
   * Computes the backward pass through the computation graph with the given
   * labels, using the actiations that are currently stored in the graph from
   * the last call to forward. Assumes that forward(...) has been called
   * already.
   */
  void backpropagate(const std::vector<BoltBatch>& labels);

  /**
   * Performs the foward and backward pass through the computation graph for the
   * given training batch. The benefit of calling this method over forward(...)
   * followed by backpropagate(...) is that there is no intermediate thread
   * synchronization.
   */
  void trainOnBatch(const std::vector<BoltBatch>& inputs,
                    const std::vector<BoltBatch>& labels);

  void trainOnBatch(const BoltBatch& inputs, const BoltBatch& labels);

  /**
   * Updates the parameters of all ops.
   */
  void updateParameters(float learning_rate);

  const std::vector<ops::OpPtr>& ops() const;

 private:
  /**
   * Helper method for trainOnBatch.
   */
  void trainOnBatchImpl(uint32_t input_batch_size, uint32_t label_batch_size);

  /**
   * Computes the forward pass through the computation graph for the given
   * sample in the batch. Assumes that setInputs(...) has already been called.
   */
  void forward(uint32_t index_in_batch);

  /**
   * Computes the backward pass through the computation graph for the given
   * sample in the batch. Assumes that setInputs(...) and setLabels(...) have
   * already been called.
   */
  void backpropagate(uint32_t index_in_batch);

  /**
   * Sets the given batch as the inputs to the computation graph.
   */
  uint32_t setInputs(const std::vector<BoltBatch>& input_batches);

  /**
   * Sets the given labels as the current labels for the computation graph.
   */
  uint32_t setLabels(const std::vector<BoltBatch>& label_batches);

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
   * These methods perform checks to make sure that computation graph is valid.
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

}  // namespace thirdai::bolt::nn::computation_graph