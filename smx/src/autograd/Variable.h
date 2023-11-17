#pragma once

#include <smx/src/tensor/Tensor.h>
#include <functional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::smx {

class Variable;
using VariablePtr = std::shared_ptr<Variable>;

using GradFunc =
    std::function<void(const TensorPtr&, const std::vector<VariablePtr>&)>;

class Variable {
 public:
  Variable(TensorPtr tensor, GradFunc grad_func,
           std::vector<VariablePtr> inputs)
      : _tensor(std::move(tensor)),
        _grad_func(std::move(grad_func)),
        _inputs(std::move(inputs)),
        _requires_grad(true) {
    if (!_inputs.empty() && !_grad_func) {
      // Error
    }
  }

  Variable(TensorPtr tensor, bool requires_grad)
      : _tensor(std::move(tensor)), _requires_grad(requires_grad) {}

  const TensorPtr& tensor() const { return _tensor; }

  bool requiresGrad() const { return _requires_grad; }

  void backpropagate() {
    TensorPtr ones;  // Ones like output.
    backpropagate(ones);
  }

  void backpropagate(const TensorPtr& grad) {
    addGradient(grad);

    auto out_degrees = outDegrees();
    // Check this has degree 0.

    std::unordered_set<Variable*> visited;
    std::queue<Variable*> queue;
    queue.push(this);

    while (!queue.empty()) {
      Variable* head = queue.front();

      if (head->_grad_func) {
        head->_grad_func(head->_grad, head->_inputs);
      }

      for (const auto& input : head->_inputs) {
        out_degrees.at(input.get())--;
        if (out_degrees.at(input.get()) == 0) {
          queue.push(input.get());
        }
      }
    }
  }

  void addGradient(const TensorPtr& grad) {
    if (!_grad) {
      _grad = grad;
    }

    // add(_grad, grad)
  }

 private:
  std::unordered_map<Variable*, size_t> outDegrees() {
    std::unordered_map<Variable*, size_t> out_degrees;

    std::unordered_set<Variable*> visited;
    std::queue<Variable*> queue;
    queue.push(this);

    while (!queue.empty()) {
      Variable* head = queue.front();
      if (!visited.count(head)) {
        visited.insert(head);

        for (const auto& input : head->_inputs) {
          out_degrees[input.get()]++;
          queue.push(input.get());
        }
      }
    }

    return out_degrees;
  }

  TensorPtr _tensor;

  GradFunc _grad_func;
  TensorPtr _grad;
  std::vector<VariablePtr> _inputs;
  bool _requires_grad;
};

}  // namespace thirdai::smx