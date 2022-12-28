#include "ComputationGraph.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::computation_graph {

ComputationGraph::ComputationGraph(
    std::vector<tensor::InputTensorPtr> inputs,
    std::vector<tensor::ActivationTensorPtr> outputs,
    std::vector<loss::LossPtr> losses)
    : _inputs(std::move(inputs)),
      _outputs(std::move(outputs)),
      _losses(std::move(losses)),
      _activations({}),
      _train_steps(0) {
  for (const auto& loss : _losses) {
    _label_inputs.push_back(loss->labels());
  }

  createOpSchedule();

  checkNoOutputsHaveDependentOps();
  checkOnlyOutputsHaveNoDependentOps();
  checkAllOutputsAreUsedInLosses();
}

void ComputationGraph::forward(const std::vector<BoltBatch>& inputs,
                               bool use_sparsity) {
  uint32_t batch_size = setInputs(inputs);

  _activations.reallocateForBatch(batch_size, use_sparsity);

  // #pragma omp parallel for default(none) shared(inputs, use_sparsity)
  for (uint32_t index_in_batch = 0; index_in_batch < batch_size;
       index_in_batch++) {
    forward(index_in_batch);
  }
}

void ComputationGraph::backpropagate(const std::vector<BoltBatch>& labels) {
  uint32_t batch_size = setLabels(labels);

  if (batch_size != _activations.currentBatchSize()) {
    throw std::invalid_argument(
        "Label batch size does not match input batch size.");
  }

  for (uint32_t index_in_batch = 0; index_in_batch < batch_size;
       index_in_batch++) {
    backpropagate(index_in_batch);
  }
}

void ComputationGraph::trainOnBatch(const std::vector<BoltBatch>& inputs,
                                    const std::vector<BoltBatch>& labels) {
  uint32_t input_batch_size = setInputs(inputs);
  uint32_t label_batch_size = setLabels(labels);

  if (input_batch_size != label_batch_size) {
    throw std::invalid_argument(
        "Input batch size and label batch size do not match.");
  }
  _activations.reallocateForBatch(input_batch_size, /* use_sparsity= */ true);

  for (uint32_t index_in_batch = 0;
       index_in_batch < inputs.at(0).getBatchSize(); index_in_batch++) {
    forward(index_in_batch);
    backpropagate(index_in_batch);
  }
}

void ComputationGraph::updateParameters(float learning_rate) {
  for (auto& op : _op_schedule) {
    op->updateParameters(learning_rate, ++_train_steps);
  }
}

void ComputationGraph::forward(uint32_t index_in_batch) {
  for (auto& op : _op_schedule) {
    op->forward(index_in_batch);
  }
}

void ComputationGraph::backpropagate(uint32_t index_in_batch) {
  _activations.resetOutputGradients(index_in_batch);

  for (auto& loss : _losses) {
    loss->computeGradients(index_in_batch);
  }

  for (auto op = _op_schedule.rbegin(); op != _op_schedule.rend(); ++op) {
    (*op)->backpropagate(index_in_batch);
  }
}

void ComputationGraph::createOpSchedule() {
  std::unordered_map<ops::OpPtr, uint32_t> in_degrees = getInDegrees();

  std::queue<ops::OpPtr> queue;

  for (const auto& input : _inputs) {
    for (const auto& op : input->dependantOps()) {
      in_degrees[op]--;
      if (in_degrees[op] == 0) {
        queue.push(op);
        in_degrees.erase(op);
      }
    }
  }

  std::vector<tensor::ActivationTensorPtr> activations;

  while (!queue.empty()) {
    auto next_op = queue.front();
    queue.pop();
    _op_schedule.push_back(next_op);

    for (const auto& output : next_op->outputs()) {
      activations.push_back(output);
      for (const auto& op : output->dependantOps()) {
        in_degrees[op]--;
        if (in_degrees[op] == 0) {
          queue.push(op);
          in_degrees.erase(op);
        }
      }
    }
  }

  _activations = ActivationsManager(activations);
}

std::unordered_map<ops::OpPtr, uint32_t> ComputationGraph::getInDegrees()
    const {
  std::unordered_map<ops::OpPtr, uint32_t> in_degrees;

  std::vector<tensor::TensorPtr> unexplored(_inputs.begin(), _inputs.end());

  while (!unexplored.empty()) {
    std::vector<tensor::TensorPtr> next_unexplored;

    for (const auto& tensor : unexplored) {
      for (const auto& op : tensor->dependantOps()) {
        if (!in_degrees.count(op)) {
          in_degrees[op] = op->outputs().size();

          next_unexplored.insert(next_unexplored.end(), op->outputs().begin(),
                                 op->outputs().end());
        }
      }
    }

    unexplored = next_unexplored;
  }

  return in_degrees;
}

void ComputationGraph::checkNoOutputsHaveDependentOps() const {
  for (const auto& output : _outputs) {
    if (!output->dependantOps().empty()) {
      throw std::invalid_argument("Outputs must not be inputs to any ops.");
    }
  }
}

void ComputationGraph::checkOnlyOutputsHaveNoDependentOps() const {
  std::unordered_set<tensor::ActivationTensorPtr> outputs_set(_outputs.begin(),
                                                              _outputs.end());

  for (const auto& activation : _activations.activationTensors()) {
    if (activation->dependantOps().empty() && !outputs_set.count(activation)) {
      throw std::invalid_argument(
          "All non outputs must be used in at least one op.");
    }
  }
}

void ComputationGraph::checkAllOutputsAreUsedInLosses() const {
  std::unordered_set<tensor::ActivationTensorPtr> outputs_set(_outputs.begin(),
                                                              _outputs.end());

  for (const auto& loss : _losses) {
    for (const auto& output : loss->outputsUsed()) {
      if (!outputs_set.count(output)) {
        throw std::invalid_argument("Only outputs can be used in losses.");
      }

      outputs_set.erase(output);
    }
  }

  if (!outputs_set.empty()) {
    throw std::invalid_argument("All outputs must be used by a loss.");
  }
}

}  // namespace thirdai::bolt::nn::computation_graph