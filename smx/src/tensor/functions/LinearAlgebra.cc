#include "dnnl.h"
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <smx/src/tensor/Tensor.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>

#ifdef DNNL_LINEAR
#include "dnnl.h"
#endif

namespace thirdai::smx {

TensorPtr add(const TensorPtr& a, const TensorPtr& b) {
  // TODO(Nicholas): add support for broadcasting.
  CHECK(a->shape() == b->shape(), "Cannot add tensors with different shapes.")
  CHECK(a->dtype() == Dtype::f32, "Can only add f32 tensors.");
  CHECK(b->dtype() == Dtype::f32, "Can only add f32 tensors.");

  if (a->isSparse() || b->isSparse()) {
    throw std::runtime_error("Adding sparse tensors is not yet supported.");
  }

  auto out = DenseTensor::make(a->shape(), Dtype::f32);

  out->eigenArray<float>() =
      dense(a)->eigenArray<float>() + dense(b)->eigenArray<float>();

  return out;
}

// TODO(Nicholas): Implement these kernels using dnnl_sgemm
DenseTensorPtr denseLinear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                           const DenseTensorPtr& b) {
  CHECK(x->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(b->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->ndim() == 2, "Weight matrix must be 2D.");
  CHECK(b->ndim() == 1, "Bias must be 1D.");
  CHECK(x->shape().last() == w->shape().last(), "Cols of x and w must match.");
  CHECK(w->shape(0) == b->shape(0), "Rows of w and b must match.");

  auto X = x->eigenMatrix<float>();
  auto W = w->eigenMatrix<float>();
  auto B = b->eigenVector<float>();

  auto out_shape = x->shape().vector();
  out_shape.back() = W.rows();
  auto out = DenseTensor::make(Shape(std::move(out_shape)), Dtype::f32);

  auto Y = out->eigenMatrix<float>();

#ifdef DNNL_LINEAR
  int64_t M = X.rows();
  int64_t N = W.rows();
  int64_t K = W.cols();

  dnnl_sgemm(/*TransA=*/'N', /*TransB=*/'T', /*M=*/M, /*N=*/N,
             /*K=*/K, /*alpha=*/1, /*a=*/X.data(), /*lda=*/K, /*b=*/W.data(),
             /*ldb=*/K, /*beta=*/0.0, Y.data(), /*ldc=*/N);
#else
  Y.noalias() = X * W.transpose();
#endif

  Y.rowwise() += B;

  return out;
}

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> denseLinearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const DenseTensorPtr& y_grad, bool compute_x_grad) {
  auto X = x->eigenMatrix<float>();
  auto W = w->eigenMatrix<float>();
  auto Y_grad = dense(y_grad)->eigenMatrix<float>();

  auto w_grad = DenseTensor::make(w->shape(), Dtype::f32);
  auto b_grad = DenseTensor::make(b->shape(), Dtype::f32);

  auto W_grad = w_grad->eigenMatrix<float>();
  auto B_grad = b_grad->eigenMatrix<float>();

#ifdef DNNL_LINEAR
  int64_t M = X.rows();
  int64_t N = W.rows();
  int64_t K = W.cols();

  dnnl_sgemm(/*TransA=*/'T', /*TransB=*/'N', /*M=*/N, /*N=*/K, /*K=*/M,
             /*alpha=*/1, /*a=*/Y_grad.data(), /*lda=*/N, /*b=*/X.data(),
             /*ldb=*/K, /*beta=*/0.0, W_grad.data(), /*ldc=*/K);
#else
  W_grad = Y_grad.transpose() * X;
#endif

  B_grad = Y_grad.colwise().sum();

  if (compute_x_grad) {
    auto x_grad = DenseTensor::make(x->shape(), Dtype::f32);
    auto X_grad = x_grad->eigenMatrix<float>();

#ifdef DNNL_LINEAR
    dnnl_sgemm(/*TransA=*/'N', /*TransB=*/'N', /*M=*/M, /*N=*/K, /*K=*/N,
               /*alpha=*/1, /*a=*/Y_grad.data(), /*lda=*/N, /*b=*/W.data(),
               /*ldb=*/K, /*beta=*/0.0, X_grad.data(), /*ldc=*/K);
#else
    X_grad = Y_grad * W;
#endif

    return {x_grad, w_grad, b_grad};
  }

  return {nullptr, w_grad, b_grad};
}

DenseTensorPtr sparseLinear(const CsrTensorPtr& x, const DenseTensorPtr& w,
                            const DenseTensorPtr& b) {
  size_t rows = x->nRows();
  size_t input_dim = w->shape(1);
  size_t dim = w->shape(0);

  auto y = DenseTensor::make(Shape(rows, dim), Dtype::f32);
  float* y_ptr = y->data<float>();

  const uint32_t* x_offsets = x->rowOffsets()->data<uint32_t>();
  const uint32_t* x_indices = x->colIndices()->data<uint32_t>();
  const float* x_values = x->colValues()->data<float>();

  const float* w_ptr = w->data<float>();
  const float* b_ptr = b->data<float>();

#pragma omp parallel for default(none) shared( \
    rows, dim, input_dim, x_offsets, x_indices, x_values, w_ptr, b_ptr, y_ptr)
  for (size_t m = 0; m < rows; m++) {
    size_t start = x_offsets[m], end = x_offsets[m + 1];

    for (size_t n = 0; n < dim; n++) {
      float act = b_ptr[n];
      for (size_t k = start; k < end; k++) {
        act += w_ptr[n * input_dim + x_indices[k]] * x_values[k];
      }
      y_ptr[m * dim + n] = act;
    }
  }

  return y;
}

std::tuple<CsrTensorPtr, DenseTensorPtr, DenseTensorPtr> sparseLinearGrad(
    const CsrTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const DenseTensorPtr& y_grad, bool compute_x_grad) {
  size_t rows = x->nRows();
  size_t input_dim = w->shape(1);
  size_t dim = w->shape(0);

  const float* y_grad_ptr = y_grad->data<float>();

  // Bias gradient
  auto b_grad = DenseTensor::make(b->shape(), Dtype::f32);
  b_grad->eigenMatrix<float>() = y_grad->eigenMatrix<float>().colwise().sum();

  // Weight gradient
  auto w_grad = DenseTensor::make(w->shape(), Dtype::f32);
  float* w_grad_ptr = w_grad->data<float>();

  const uint32_t* x_offsets = x->rowOffsets()->data<uint32_t>();
  const uint32_t* x_indices = x->colIndices()->data<uint32_t>();
  const float* x_values = x->colValues()->data<float>();

#pragma omp parallel for default(none)                                       \
    shared(rows, dim, input_dim, x_offsets, x_indices, x_values, y_grad_ptr, \
           w_grad_ptr)
  for (size_t n = 0; n < dim; n++) {
    float* w_n_grad = w_grad_ptr + n * input_dim;
    std::fill(w_n_grad, w_n_grad + input_dim, 0);
    for (size_t m = 0; m < rows; m++) {
      size_t start = x_offsets[m], end = x_offsets[m + 1];
      for (size_t k = start; k < end; k++) {
        w_n_grad[x_indices[k]] += y_grad_ptr[m * dim + n] * x_values[k];
      }
    }
  }

  if (!compute_x_grad) {
    return {nullptr, w_grad, b_grad};
  }

  // Input gradient
  const float* w_ptr = w->data<float>();

  auto x_grad = DenseTensor::make(x->colValues()->shape(), Dtype::f32);
  float* x_grad_ptr = x_grad->data<float>();

#pragma omp parallel for default(none) shared( \
    rows, dim, input_dim, x_offsets, x_indices, x_grad_ptr, y_grad_ptr, w_ptr)
  for (size_t m = 0; m < rows; m++) {
    size_t start = x_offsets[m], end = x_offsets[m + 1];
    std::fill(x_grad_ptr + start, x_grad_ptr + end, 0);
    for (size_t n = 0; n < dim; n++) {
      const float* w_n = w_ptr + n * input_dim;
      float y_m_n_grad = y_grad_ptr[m * dim + n];
      for (size_t k = start; k < end; k++) {
        x_grad_ptr[k] += y_m_n_grad * w_n[x_indices[k]];
      }
    }
  }

  return {CsrTensor::make(x->rowOffsets(), x->colIndices(), x_grad, x->shape()),
          w_grad, b_grad};
}

DenseTensorPtr linear(const TensorPtr& x, const DenseTensorPtr& w,
                      const DenseTensorPtr& b) {
  CHECK(x->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(b->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->ndim() == 2, "Weight matrix must be 2D.");
  CHECK(b->ndim() == 1, "Bias must be 1D.");
  CHECK(x->shape().last() == w->shape().last(), "Cols of x and w must match.");
  CHECK(w->shape(0) == b->shape(0), "Rows of w and b must match.");

  if (x->isSparse()) {
    return sparseLinear(csr(x), w, b);
  }

  return denseLinear(dense(x), w, b);
}

std::tuple<TensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const TensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const DenseTensorPtr& y_grad, bool compute_x_grad) {
  if (x->isSparse()) {
    return sparseLinearGrad(csr(x), w, b, y_grad, compute_x_grad);
  }

  return denseLinearGrad(dense(x), w, b, y_grad, compute_x_grad);
}

}  // namespace thirdai::smx