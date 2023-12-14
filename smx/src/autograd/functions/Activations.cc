#include "Activations.h"
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr relu(const VariablePtr& in) {
  const auto& tensor = in->tensor();

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented relu(sparse).");
  }

  auto out = relu(dense(tensor));

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = reluGrad(out, dense(out_grad));

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr tanh(const VariablePtr& in) {
  const auto& tensor = in->tensor();

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented relu(sparse).");
  }

  auto out = tanh(dense(tensor));

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = tanhGrad(out, dense(out_grad));

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr sigmoid(const VariablePtr& in) {
  const auto& tensor = in->tensor();

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented relu(sparse).");
  }

  auto out = sigmoid(dense(tensor));

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = sigmoidGrad(out, dense(out_grad));

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

VariablePtr softmax(const VariablePtr& in) {
  const auto& tensor = in->tensor();

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented relu(sparse).");
  }

  auto out = softmax(dense(tensor));

  GradFunc grad_func = [out](const TensorPtr& out_grad,
                             const std::vector<VariablePtr>& inputs) {
    if (!inputs.at(0)->requiresGrad()) {
      return;
    }

    auto in_grad = softmaxGrad(out, dense(out_grad));

    inputs.at(0)->addGradient(in_grad);
  };

  return Variable::make(out, grad_func, {in});
}

}  // namespace thirdai::smx