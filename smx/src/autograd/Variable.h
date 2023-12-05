#pragma once

#include <smx/src/tensor/Tensor.h>
#include <functional>
#include <queue>
#include <stdexcept>
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

  static auto make(TensorPtr tensor, GradFunc grad_func,
                   std::vector<VariablePtr> inputs) {
    return std::make_shared<Variable>(std::move(tensor), grad_func,
                                      std::move(inputs));
  }

  static auto make(TensorPtr tensor, bool requires_grad) {
    return std::make_shared<Variable>(std::move(tensor), requires_grad);
  }

  const TensorPtr& tensor() const { return _tensor; }

  bool requiresGrad() const { return _requires_grad; }

  void backpropagate() {
    TensorPtr ones;  // Ones like output.
    backpropagate(ones);
  }

  void backpropagate(const TensorPtr& grad);

  void addGradient(const TensorPtr& grad) {
    if (grad->shape() != _tensor->shape()) {
      throw std::invalid_argument(
          "Cannot assign gradient with shape " + grad->shape().toString() +
          " to variable with shape " + _tensor->shape().toString() + ".");
    }

    if (!_grad) {
      _grad = grad;
    }

    // _grad = add(_grad, grad)
  }

 private:
  std::vector<Variable*> topologicalSort();

  TensorPtr _tensor;

  GradFunc _grad_func;
  TensorPtr _grad;
  std::vector<VariablePtr> _inputs;
  bool _requires_grad;
};

}  // namespace thirdai::smx