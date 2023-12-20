#include "Activations.h"
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr relu(const VariablePtr& in) {
  const auto& tensor = in->tensor();

  if (tensor->isSparse()) {
    auto in_csr = csr(tensor);
    auto out = CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                               relu(in_csr->colValues()), in_csr->shape());

    GradFunc grad_func = [out](const TensorPtr& out_grad,
                               const std::vector<VariablePtr>& inputs) {
      if (!inputs.at(0)->requiresGrad()) {
        return;
      }

      auto in_grad = CsrTensor::make(
          out->rowOffsets(), out->colIndices(),
          reluGrad(out->colValues(), csr(out_grad)->colValues()), out->shape());

      inputs.at(0)->addGradient(in_grad);
    };

    return Variable::make(out, grad_func, {in});
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
    auto in_csr = csr(tensor);
    auto out = CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                               tanh(in_csr->colValues()), in_csr->shape());

    GradFunc grad_func = [out](const TensorPtr& out_grad,
                               const std::vector<VariablePtr>& inputs) {
      if (!inputs.at(0)->requiresGrad()) {
        return;
      }

      auto in_grad = CsrTensor::make(
          out->rowOffsets(), out->colIndices(),
          tanhGrad(out->colValues(), csr(out_grad)->colValues()), out->shape());

      inputs.at(0)->addGradient(in_grad);
    };

    return Variable::make(out, grad_func, {in});
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
    auto in_csr = csr(tensor);
    auto out = CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                               sigmoid(in_csr->colValues()), in_csr->shape());

    GradFunc grad_func = [out](const TensorPtr& out_grad,
                               const std::vector<VariablePtr>& inputs) {
      if (!inputs.at(0)->requiresGrad()) {
        return;
      }

      auto in_grad = CsrTensor::make(
          out->rowOffsets(), out->colIndices(),
          sigmoidGrad(out->colValues(), csr(out_grad)->colValues()),
          out->shape());

      inputs.at(0)->addGradient(in_grad);
    };

    return Variable::make(out, grad_func, {in});
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
    auto out = softmax(csr(tensor));

    GradFunc grad_func = [out](const TensorPtr& out_grad,
                               const std::vector<VariablePtr>& inputs) {
      if (!inputs.at(0)->requiresGrad()) {
        return;
      }

      auto in_grad = softmaxGrad(out, csr(out_grad));

      inputs.at(0)->addGradient(in_grad);
    };

    return Variable::make(out, grad_func, {in});
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