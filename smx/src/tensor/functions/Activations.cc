#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <cmath>
#include <sstream>
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

TensorPtr relu(const TensorPtr& in) {
  if (in->isSparse()) {
    auto in_csr = csr(in);
    return CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                           relu(in_csr->colValues()), in_csr->shape());
  }
  return relu(dense(in));
}

TensorPtr reluGrad(const TensorPtr& out, const TensorPtr& out_grad) {
  if (out->isSparse()) {
    auto out_csr = csr(out);
    auto out_grad_csr = csr(out_grad);
    return CsrTensor::make(
        out_csr->rowOffsets(), out_csr->colIndices(),
        reluGrad(out_csr->colValues(), out_grad_csr->colValues()),
        out_csr->shape());
  }

  return reluGrad(dense(out), dense(out_grad));
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

TensorPtr tanh(const TensorPtr& in) {
  if (in->isSparse()) {
    auto in_csr = csr(in);
    return CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                           tanh(in_csr->colValues()), in_csr->shape());
  }
  return tanh(dense(in));
}

TensorPtr tanhGrad(const TensorPtr& out, const TensorPtr& out_grad) {
  if (out->isSparse()) {
    auto out_csr = csr(out);
    auto out_grad_csr = csr(out_grad);
    return CsrTensor::make(
        out_csr->rowOffsets(), out_csr->colIndices(),
        tanhGrad(out_csr->colValues(), out_grad_csr->colValues()),
        out_csr->shape());
  }

  return tanhGrad(dense(out), dense(out_grad));
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

TensorPtr sigmoid(const TensorPtr& in) {
  if (in->isSparse()) {
    auto in_csr = csr(in);
    return CsrTensor::make(in_csr->rowOffsets(), in_csr->colIndices(),
                           sigmoid(in_csr->colValues()), in_csr->shape());
  }
  return sigmoid(dense(in));
}

TensorPtr sigmoidGrad(const TensorPtr& out, const TensorPtr& out_grad) {
  if (out->isSparse()) {
    auto out_csr = csr(out);
    auto out_grad_csr = csr(out_grad);
    return CsrTensor::make(
        out_csr->rowOffsets(), out_csr->colIndices(),
        sigmoidGrad(out_csr->colValues(), out_grad_csr->colValues()),
        out_csr->shape());
  }

  return sigmoidGrad(dense(out), dense(out_grad));
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

CsrTensorPtr softmax(const CsrTensorPtr& in) {
  CHECK(in->dtype() == Dtype::f32,
        "Softmax can only be applied to f32 tensors.");
  size_t n_rows = in->nRows();

  const uint32_t* offsets = in->rowOffsets()->data<uint32_t>();
  const float* x = in->colValues()->data<float>();

  auto out_values = DenseTensor::make(in->colValues()->shape(), Dtype::f32);
  float* y = out_values->data<float>();

#pragma omp parallel for default(none) shared(n_rows, offsets, x, y)
  for (size_t n = 0; n < n_rows; n++) {
    size_t start = offsets[n];
    size_t end = offsets[n + 1];

    float max = std::numeric_limits<float>::lowest();
    for (size_t i = start; i < end; i++) {
      if (x[i] > max) {
        max = x[i];
      }
    }
    float sum = 0.0;
    for (size_t i = start; i < end; i++) {
      y[i] = std::exp(x[i] - max);
      sum += y[i];
    }
    for (size_t i = start; i < end; i++) {
      y[i] /= sum;
    }
  }

  return CsrTensor::make(in->rowOffsets(), in->colIndices(), out_values,
                         in->shape());
}

CsrTensorPtr softmaxGrad(const CsrTensorPtr& out,
                         const CsrTensorPtr& out_grad) {
  CHECK(out->dtype() == Dtype::f32,
        "Activations are only supported for f32 tensors.")
  CHECK(out_grad->dtype() == Dtype::f32,
        "Activations are only supported for f32.")
  CHECK(out->shape() == out_grad->shape(),
        "Output and grad shapes must match.");

  const uint32_t* offsets = out->rowOffsets()->data<uint32_t>();
  const float* y = out->colValues()->data<float>();
  const float* y_grad = out_grad->colValues()->data<float>();

  auto in_grad = DenseTensor::make(out->colValues()->shape(), Dtype::f32);

  float* x_grad = in_grad->data<float>();

  size_t n_rows = out->nRows();

#pragma omp parallel for default(none) \
    shared(n_rows, offsets, y, y_grad, x_grad)
  for (size_t n = 0; n < n_rows; n++) {
    size_t start = offsets[n];
    size_t end = offsets[n + 1];

    float sum_gy = 0.0;
    for (size_t i = start; i < end; i++) {
      sum_gy += y[i] * y_grad[i];
    }

    for (size_t i = start; i < end; i++) {
      x_grad[i] = y[i] * (y_grad[i] - sum_gy) * y_grad[i];
    }
  }

  return CsrTensor::make(out->rowOffsets(), out->colIndices(), in_grad,
                         out->shape());
}

TensorPtr softmax(const TensorPtr& in) {
  if (in->isSparse()) {
    return softmax(csr(in));
  }
  return softmax(dense(in));
}

TensorPtr softmaxGrad(const TensorPtr& out, const TensorPtr& out_grad) {
  if (out->isSparse()) {
    return softmaxGrad(csr(out), csr(out_grad));
  }

  return softmaxGrad(dense(out), dense(out_grad));
}

}  // namespace thirdai::smx