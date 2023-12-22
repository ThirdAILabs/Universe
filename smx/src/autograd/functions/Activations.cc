#include "Activations.h"
#include <smx/src/tensor/Functions.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr relu(const VariablePtr& in) {
  auto out = relu(in->tensor());

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = reluGrad(out, out_grad);

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr tanh(const VariablePtr& in) {
  auto out = tanh(in->tensor());

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = tanhGrad(out, out_grad);

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr sigmoid(const VariablePtr& in) {
  auto out = sigmoid(in->tensor());

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = sigmoidGrad(out, out_grad);

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr softmax(const VariablePtr& in) {
  auto out = softmax(in->tensor());

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = softmaxGrad(out, out_grad);

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

}  // namespace thirdai::smx