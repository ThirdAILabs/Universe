#include "ComputationGraph.h"
#include <bolt/src/nn/ops/Op.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt {

ComputationList getComputationOrder(const ComputationList& inputs,
                                    const ComputationList& outputs,
                                    const std::vector<LossPtr>& losses) {
  checkLossesOnlyApplyToTerminalOutputs(losses);

  auto out_degrees = countDependentComputations(losses);

  std::queue<ComputationPtr> queue;

  for (const auto& output : computationsUsedInLossFunctions(losses)) {
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
            "Model computation depends on an input that is not present in the "
            "list of inputs to the model.");
      }
      inputs_set.erase(*comp);
    } else {
      computation_order.push_back(*comp);
    }
  }

  if (!inputs_set.empty()) {
    size_t input_idx = std::distance(
        inputs.begin(),
        std::find(inputs.begin(), inputs.end(), *inputs_set.begin()));

    throw std::invalid_argument(
        "The input passed at index " + std::to_string(input_idx) +
        " was not used by any computation in the model.");
  }

  checkAllOutputsInComputationOrder(/* computation_order= */ computation_order,
                                    /* outputs= */ outputs);

  return computation_order;
}

std::unordered_map<ComputationPtr, uint32_t> countDependentComputations(
    const std::vector<LossPtr>& losses) {
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

  for (const auto& output : computationsUsedInLossFunctions(losses)) {
    recurse(output);
  }

  return out_degrees;
}

ComputationList computationsUsedInLossFunctions(
    const std::vector<LossPtr>& losses) {
  std::unordered_set<ComputationPtr> comps;

  for (const auto& loss : losses) {
    for (const auto& output : loss->outputsUsed()) {
      if (comps.count(output)) {
        throw std::invalid_argument(
            "Two loss functions cannot be applied to the same computation.");
      }
      comps.insert(output);
    }
  }

  return {comps.begin(), comps.end()};
}

void checkLossesOnlyApplyToTerminalOutputs(const std::vector<LossPtr>& losses) {
  auto out_degrees = countDependentComputations(losses);

  for (const auto& output : computationsUsedInLossFunctions(losses)) {
    if (out_degrees.count(output)) {
      throw std::invalid_argument(
          "Computations used in loss functions must not be inputs to any "
          "further ops. Found computation that is used in a loss function and "
          "as an input to another computation.");
    }
  }
}

void checkAllOutputsInComputationOrder(const ComputationList& computation_order,
                                       const ComputationList& outputs) {
  for (const auto& output : outputs) {
    if (std::find(computation_order.begin(), computation_order.end(), output) ==
        computation_order.end()) {
      throw std::invalid_argument(
          "Model contains an output that is not found in the computation graph "
          "created from traversing backward from the specified loss "
          "functions.");
    }
  }
}

}  // namespace thirdai::bolt