#include "Variable.h"
#include <unordered_set>

namespace thirdai::smx {

void Variable::backpropagate(const TensorPtr& grad) {
  addGradient(grad);

  auto topo_order = topologicalSort();

  for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
    Variable* var = *it;

    if (var->_grad_func) {
      var->_grad_func(var->_grad, var->_inputs);
    }
  }
}

std::vector<Variable*> Variable::topologicalSort() {
  std::vector<Variable*> sorted;
  std::unordered_set<Variable*> visited;

  std::function<void(Variable*)> traverse;

  traverse = [&traverse, &sorted, &visited](Variable* curr) {
    if (visited.count(curr)) {
      return;
    }

    for (auto& input : curr->_inputs) {
      traverse(input.get());
    }

    visited.insert(curr);
    sorted.push_back(curr);
  };

  traverse(this);

  return sorted;
}

}  // namespace thirdai::smx