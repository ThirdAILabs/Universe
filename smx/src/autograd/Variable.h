#pragma once

#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Tensor.h>
#include <functional>
#include <stdexcept>
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

  const TensorPtr& grad() const { return _grad; }

  bool requiresGrad() const { return _requires_grad; }

  void backward() {
    TensorPtr ones;  // Ones like output.
    backward(ones);
  }

  void backward(const TensorPtr& grad);

  void addGradient(const TensorPtr& grad);

 private:
  std::vector<Variable*> topologicalSort();

  TensorPtr _tensor;

  GradFunc _grad_func;
  TensorPtr _grad;
  std::vector<VariablePtr> _inputs;
  bool _requires_grad;
};

}  // namespace thirdai::smx