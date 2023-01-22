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

  ComputationList computation_order_rev;

  while (!queue.empty()) {
    auto next_computation = queue.front();
    queue.pop();
    computation_order_rev.push_back(next_computation);

    for (const auto& input : next_computation->inputs()) {
      out_degrees.at(input)--;
      if (out_degrees.at(input) == 0) {
        queue.push(input);
        out_degrees.erase(input);
      }
    }
  }

  std::unordered_set<ComputationPtr> inputs_set(inputs.begin(), inputs.end());

  ComputationList computation_order;

  for (auto comp = computation_order_rev.rbegin();
       comp != computation_order_rev.rend(); ++comp) {
    // Remove all input computations from the final computation order.
    if ((*comp)->inputs().empty()) {
      if (!inputs_set.count(*comp)) {
        throw std::invalid_argument(
            "Model computation depends on input '" + (*comp)->name() +
            "' that is not present in the list of inputs to the model.");
      }
      inputs_set.erase(*comp);
    } else {
      computation_order.push_back(*comp);
    }
  }

  if (!inputs_set.empty()) {
    throw std::invalid_argument(
        "Input '" + (*inputs_set.begin())->name() +
        "' was not used by any computation in the model.");
  }

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