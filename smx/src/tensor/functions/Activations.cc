#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <stdexcept>

namespace thirdai::smx {

DenseTensorPtr relu(const DenseTensorPtr& in) {
  CHECK(in->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")

  auto out = DenseTensor::make(in->shape(), Dtype::f32);
  out->eigenArray<float>() = in->eigenArray<float>().max(0.0);

  return out;
}

DenseTensorPtr reluGrad(const DenseTensorPtr& out,
                        const DenseTensorPtr& out_grad) {
  CHECK(out->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")
  CHECK(out_grad->dtype() == Dtype::f32,
        "Activations are only supported for f32.")

  auto in_grad = DenseTensor::make(out->shape(), Dtype::f32);

  const float* out_ptr = out->data<float>();
  const float* out_grad_ptr = out_grad->data<float>();
  float* in_grad_ptr = in_grad->data<float>();

  size_t N = out->size();
  // TODO(Nicholas) is 1000 an ok threshold for parallelism?
#pragma omp parallel for default(none) \
    shared(in_grad_ptr, out_ptr, out_grad_ptr, N) if (N > 1000)
  for (size_t i = 0; i < N; i++) {
    in_grad_ptr[i] = out_ptr[i] > 0 ? out_grad_ptr[i] : 0;
  }

  return in_grad;
}

DenseTensorPtr tanh(const DenseTensorPtr& in) {
  CHECK(in->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")

  auto out = DenseTensor::make(in->shape(), Dtype::f32);
  out->eigenArray<float>() = in->eigenArray<float>().tanh();

  return out;
}

DenseTensorPtr tanhGrad(const DenseTensorPtr& out,
                        const DenseTensorPtr& out_grad) {
  CHECK(out->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")
  CHECK(out_grad->dtype() == Dtype::f32,
        "Activations are only supported for f32.")

  auto in_grad = DenseTensor::make(out->shape(), Dtype::f32);

  in_grad->eigenArray<float>() =
      (1 - out->eigenArray<float>().square()) * out_grad->eigenArray<float>();

  return in_grad;
}

DenseTensorPtr sigmoid(const DenseTensorPtr& in) {
  CHECK(in->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")

  auto out = DenseTensor::make(in->shape(), Dtype::f32);
  out->eigenArray<float>() = (1 + (-in->eigenArray<float>()).exp()).inverse();

  return out;
}

DenseTensorPtr sigmoidGrad(const DenseTensorPtr& out,
                           const DenseTensorPtr& out_grad) {
  CHECK(out->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")
  CHECK(out_grad->dtype() == Dtype::f32,
        "Activations are only supported for f32.")

  auto in_grad = DenseTensor::make(out->shape(), Dtype::f32);

  auto y = out->eigenArray<float>();
  in_grad->eigenArray<float>() =
      (y - y.square()) * out_grad->eigenArray<float>();

  return in_grad;
}

DenseTensorPtr softmax(const DenseTensorPtr& in) {
  CHECK(in->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")

  auto out = DenseTensor::make(in->shape(), Dtype::f32);

  auto x = in->eigenMatrix<float>().array();
  auto y = out->eigenMatrix<float>().array();

  auto maxes = x.rowwise().maxCoeff();
  y = (x.colwise() - maxes).exp();
  y.colwise() /= y.rowwise().sum();

  return out;
}

DenseTensorPtr softmaxGrad(const DenseTensorPtr& out,
                           const DenseTensorPtr& out_grad) {
  CHECK(out->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")
  CHECK(out_grad->dtype() == Dtype::f32,
        "Activations are only supported for f32.")

  auto in_grad = DenseTensor::make(out->shape(), Dtype::f32);

  auto y = out->eigenMatrix<float>().array();
  auto y_grad = out_grad->eigenMatrix<float>().array();

  auto gy = (y * y_grad).eval();

  in_grad->eigenMatrix<float>().array() =
      (gy - y.colwise() * gy.rowwise().sum()) * y_grad;

  return in_grad;
}

}  // namespace thirdai::smx