#include "ComputationGraph.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::bolt::nn::graph {

ComputationGraph::ComputationGraph(
    std::vector<tensor::InputTensorPtr> inputs,
    std::vector<tensor::ActivationTensorPtr> outputs)
    : _inputs(std::move(inputs)), _outputs(std::move(outputs)) {
  createOpSchedule();

  for (auto& output : _outputs) {
    if (!output->dependantOps().empty()) {
      throw std::invalid_argument("Outputs must not be inputs to any ops.");
    }
  }
}

std::unordered_map<ops::OpPtr, uint32_t> ComputationGraph::getInDegrees()
    const {
  std::unordered_map<ops::OpPtr, uint32_t> in_degrees;

  std::vector<tensor::TensorPtr> unexplored;
  unexplored.insert(unexplored.end(), _inputs.begin(), _inputs.end());

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

}  // namespace thirdai::bolt::nn::graph