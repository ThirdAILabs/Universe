#include "ComputationGraph.h"
#include <bolt/src/nn/ops/Op.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::autograd {

ComputationList getComputationOrder(const ComputationList& inputs,
                                    const ComputationList& outputs) {
  auto out_degrees = countDependentComputations(outputs);

  std::queue<ComputationPtr> queue;

  for (const auto& output : outputs) {
    queue.push(output);
  }

  ComputationList computation_order;

  while (!queue.empty()) {
    auto next_computation = queue.front();
    queue.pop();
    computation_order.push_back(next_computation);

    for (const auto& input : next_computation->inputs()) {
      out_degrees.at(input)--;
      if (out_degrees.at(input) == 0) {
        queue.push(input);
        out_degrees.erase(input);
      }
    }
  }

  std::unordered_set<ComputationPtr> inputs_set(inputs.begin(), inputs.end());
  // Remove all input computations from the final computation order.
  while (computation_order.back()->inputs().empty()) {
    if (!inputs_set.count(computation_order.back())) {
      throw std::invalid_argument(
          "Model computation depends on input '" +
          computation_order.back()->name() +
          "' that is not present in the list of inputs to the model.");
    }
    inputs_set.erase(computation_order.back());
    computation_order.pop_back();
  }
  if (!inputs_set.empty()) {
    throw std::invalid_argument(
        "Input '" + (*inputs_set.begin())->name() +
        "' was not used by any computation in the model.");
  }

  std::reverse(computation_order.begin(), computation_order.end());

  return computation_order;
}

std::unordered_map<ComputationPtr, uint32_t> countDependentComputations(
    const ComputationList& outputs) {
  std::unordered_map<ComputationPtr, uint32_t> out_degrees;

  std::unordered_set<ComputationPtr> visited;

  std::function<void(const ComputationPtr&)> recurse;

  recurse = [&visited, &out_degrees, &recurse](const ComputationPtr& comp) {
    if (visited.count(comp)) {
      return;
    }

    visited.insert(comp);

    for (const auto& input : comp->inputs()) {
      out_degrees[input]++;
      recurse(input);
    }
  };

  for (const auto& output : outputs) {
    recurse(output);
  }

  return out_degrees;
}

}  // namespace thirdai::bolt::nn::autograd