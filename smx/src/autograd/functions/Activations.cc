#include "Activations.h"
#include <smx/src/tensor/DenseTensor.h>
#include <stdexcept>

namespace thirdai::smx {

VariablePtr relu(const VariablePtr& input) {
  const auto& tensor = input->tensor();
  if (tensor->dtype() != Dtype::f32) {
    throw std::invalid_argument(
        "Activations can only be applied to f32 tensors.");
  }

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented relu(sparse).");
  }

  auto dense = asDense(tensor);

  auto output = DenseTensor::make(dense->shape(), dense->dtype());

  output->eigenArray<float>() = dense->eigenArray<float>().max(0.0);

  GradFunc grad_func = [output](const TensorPtr& output_grad,
                                const std::vector<VariablePtr>& inputs) {
    if (!inputs[0]->requiresGrad()) {
      return;
    }
    const float* output_ptr = output->data<float>();
    const float* output_grad_ptr = asDense(output_grad)->data<float>();

    auto input_grad = DenseTensor::make(output->shape(), output->dtype());
    float* input_grad_ptr = input_grad->data<float>();

    for (size_t i = 0; i < output->shape().size(); i++) {
      if (output_ptr[i] > 0) {
        input_grad_ptr[i] = output_grad_ptr[i];
      } else {
        input_grad_ptr[i] = 0;
      }
    }

    inputs[0]->addGradient(input_grad);
  };

  return Variable::make(output, grad_func, {input});
}

VariablePtr tanh(const VariablePtr& input) {
  const auto& tensor = input->tensor();
  if (tensor->dtype() != Dtype::f32) {
    throw std::invalid_argument(
        "Activations can only be applied to f32 tensors.");
  }

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented tanh(sparse).");
  }

  auto dense = asDense(tensor);

  auto output = DenseTensor::make(dense->shape(), dense->dtype());

  output->eigenArray<float>() = dense->eigenArray<float>().tanh();

  GradFunc grad_func = [output](const TensorPtr& output_grad,
                                const std::vector<VariablePtr>& inputs) {
    if (!inputs[0]->requiresGrad()) {
      return;
    }

    auto input_grad = DenseTensor::make(output->shape(), output->dtype());

    input_grad->eigenArray<float>() =
        (1 - output->eigenArray<float>().square()) *
        asDense(output_grad)->eigenArray<float>();

    inputs[0]->addGradient(input_grad);
  };

  return Variable::make(output, grad_func, {input});
}

VariablePtr sigmoid(const VariablePtr& input) {
  const auto& tensor = input->tensor();
  if (tensor->dtype() != Dtype::f32) {
    throw std::invalid_argument(
        "Activations can only be applied to f32 tensors.");
  }

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented sigmoid(sparse).");
  }

  auto dense = asDense(tensor);

  auto output = DenseTensor::make(dense->shape(), dense->dtype());

  output->eigenArray<float>() =
      (1 + (-dense->eigenArray<float>()).exp()).inverse();

  GradFunc grad_func = [output](const TensorPtr& output_grad,
                                const std::vector<VariablePtr>& inputs) {
    if (!inputs[0]->requiresGrad()) {
      return;
    }

    auto input_grad = DenseTensor::make(output->shape(), output->dtype());

    auto y = output->eigenArray<float>();
    input_grad->eigenArray<float>() =
        (y - y.square()) * asDense(output_grad)->eigenArray<float>();

    inputs[0]->addGradient(input_grad);
  };

  return Variable::make(output, grad_func, {input});
}

VariablePtr softmax(const VariablePtr& input) {
  const auto& tensor = input->tensor();
  if (tensor->dtype() != Dtype::f32) {
    throw std::invalid_argument(
        "Activations can only be applied to f32 tensors.");
  }

  if (tensor->isSparse()) {
    throw std::runtime_error("Not implemented sigmoid(sparse).");
  }

  auto dense = asDense(tensor);

  auto output = DenseTensor::make(dense->shape(), dense->dtype());

  auto in = dense->eigenMatrix<float>().array();
  auto out = output->eigenMatrix<float>().array();
  auto maxes = in.rowwise().maxCoeff();
  out = (in - maxes).exp();
  out /= out.rowwise().sum();

  GradFunc grad_func = [output](const TensorPtr& output_grad,
                                const std::vector<VariablePtr>& inputs) {
    if (!inputs[0]->requiresGrad()) {
      return;
    }

    auto in_grad = DenseTensor::make(output->shape(), output->dtype());

    auto out = output->eigenMatrix<float>().array();
    auto out_grad = asDense(output_grad)->eigenMatrix<float>().array();

    auto gy = (out * out_grad).eval();

    in_grad->eigenMatrix<float>().array() =
        (gy - out * gy.rowwise().sum()) * out_grad;

    inputs[0]->addGradient(in_grad);
  };

  return Variable::make(output, grad_func, {input});
}

}  // namespace thirdai::smx