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
      asDense(a)->eigenArray<float>() + asDense(b)->eigenArray<float>();

  return out;
}

TensorPtr linear(const TensorPtr& x, const DenseTensorPtr& w,
                 const DenseTensorPtr& b) {
  CHECK(x->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(b->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->ndim() == 2, "Weight matrix must be 2D.");
  CHECK(w->ndim() == 1, "Bias must be 1D.");

  if (x->isSparse()) {
    throw std::invalid_argument(
        "Embedding layer is recommended in place of linear with sparse input.");
  }

  auto X = asDense(x)->eigenMatrix<float>();
  auto W = w->eigenMatrix<float>();
  auto B = b->eigenVector<float>();

  CHECK(X.cols() == W.cols(), "Last dims of x and w should have same size.");
  CHECK(W.rows() == B.cols(), "Weight matrix and bias should match.");

  auto out_shape = x->shape().vector();
  out_shape.back() = W.rows();
  auto out = DenseTensor::make(Shape(std::move(out_shape)), Dtype::f32);

  auto Y = out->eigenMatrix<float>();

  Y.noalias() = X * W.transpose();
  Y.rowwise() += B;

  return out;
}

std::tuple<TensorPtr, TensorPtr, TensorPtr> linearGrad(const TensorPtr& x,
                                                       const DenseTensorPtr& w,
                                                       const DenseTensorPtr& b,
                                                       const TensorPtr& y_grad,
                                                       bool compute_x_grad) {
  if (x->isSparse()) {
    throw std::invalid_argument(
        "Embedding layer is recommended in place of linear with sparse input.");
  }

  auto X = asDense(x)->eigenMatrix<float>();
  auto W = w->eigenMatrix<float>();
  auto Y_grad = asDense(y_grad)->eigenMatrix<float>();

  auto x_grad =
      compute_x_grad ? DenseTensor::make(x->shape(), Dtype::f32) : nullptr;
  auto w_grad = DenseTensor::make(w->shape(), Dtype::f32);
  auto b_grad = DenseTensor::make(b->shape(), Dtype::f32);

  auto W_grad = w_grad->eigenMatrix<float>();
  auto B_grad = b_grad->eigenMatrix<float>();

  W_grad = Y_grad.transpose() * X;
  B_grad = Y_grad.rowwise().sum();  // TODO(Nicholas): rowwise or colwise?

  if (compute_x_grad) {
    auto X_grad = x_grad->eigenMatrix<float>();
    X_grad = Y_grad * W;
  }

  return {x_grad, w_grad, b_grad};
}

}  // namespace thirdai::smx