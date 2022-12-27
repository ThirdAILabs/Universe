#include "ComputationGraph.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::graph {

ComputationGraph::ComputationGraph(
    std::vector<tensor::InputTensorPtr> inputs,
    std::vector<tensor::ActivationTensorPtr> outputs,
    std::vector<loss::LossPtr> losses)
    : _inputs(std::move(inputs)),
      _outputs(std::move(outputs)),
      _losses(std::move(losses)) {
  createOpSchedule();

  checkNoOutputsHaveDependentOps();
  checkOnlyOutputsHaveNoDependentOps();
  checkAllOutputsAreUsedInLosses();
}

void ComputationGraph::forward(uint32_t index_in_batch) {
  for (auto& op : _op_schedule) {
    op->forward(index_in_batch);
  }
}

void ComputationGraph::backpropagate(uint32_t index_in_batch,
                                     const std::vector<BoltBatch>& labels) {
  for (uint32_t i = 0; i < _losses.size(); i++) {
    _losses[i]->computeGradients(index_in_batch, labels[i][index_in_batch]);
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

  while (!queue.empty()) {
    auto next_op = queue.front();
    queue.pop();
    _op_schedule.push_back(next_op);

    for (const auto& output : next_op->outputs()) {
      _activations.push_back(output);
      for (const auto& op : output->dependantOps()) {
        in_degrees[op]--;
        if (in_degrees[op] == 0) {
          queue.push(op);
          in_degrees.erase(op);
        }
      }
    }
  }
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

  for (const auto& activation : _activations) {
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

}  // namespace thirdai::bolt::nn::graph