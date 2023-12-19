#include <smx/src/tensor/Functions.h>
#include <cstddef>
#include <stdexcept>

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
DenseTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
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

  Y.noalias() = X * W.transpose();
  Y.rowwise() += B;

  return out;
}

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const DenseTensorPtr& y_grad, bool compute_x_grad) {
  auto X = x->eigenMatrix<float>();
  auto W = w->eigenMatrix<float>();
  auto Y_grad = dense(y_grad)->eigenMatrix<float>();

  auto w_grad = DenseTensor::make(w->shape(), Dtype::f32);
  auto b_grad = DenseTensor::make(b->shape(), Dtype::f32);

  auto W_grad = w_grad->eigenMatrix<float>();
  auto B_grad = b_grad->eigenMatrix<float>();

  W_grad = Y_grad.transpose() * X;
  B_grad = Y_grad.colwise().sum();

  if (compute_x_grad) {
    auto x_grad = DenseTensor::make(x->shape(), Dtype::f32);
    auto X_grad = x_grad->eigenMatrix<float>();
    X_grad = Y_grad * W;

    return {x_grad, w_grad, b_grad};
  }

  return {nullptr, w_grad, b_grad};
}

}  // namespace thirdai::smx