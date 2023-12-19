#include <_types/_uint32_t.h>
#include <smx/src/tensor/Functions.h>

namespace thirdai::smx {

inline void sgemsv(const float* x, const float* w, const float* b,
                   size_t input_dim, const uint32_t* y_indices, float* y_values,
                   size_t y_nonzeros) {
  for (size_t n = 0; n < y_nonzeros; n++) {
    const float* w_n = w + y_indices[n] * input_dim;

    float act = b[y_indices[n]];
#pragma omp simd
    for (size_t i = 0; i < input_dim; i++) {
      act += w_n[i] * x[i];
    }
    y_values[n] = act;
  }
}

inline void sgemsvGrad(const float* x, float* x_grad, const float* w,
                       float* w_grad, float* b_grad, size_t input_dim,
                       const uint32_t* y_indices, const float* y_grad,
                       size_t y_nonzeros) {
  for (size_t n = 0; n < y_nonzeros; n++) {
    size_t neuron = y_indices[n];
    const float* w_n = w + neuron * input_dim;
    float* w_g_n = w_grad + neuron * input_dim;

    for (size_t i = 0; i < input_dim; i++) {
      w_g_n[i] += y_grad[neuron] * x[i];
      x_grad[i] += y_grad[neuron] * w_n[i];
    }

    b_grad[neuron] += y_grad[n];
  }
}

CsrTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                    const DenseTensorPtr& b, float sparsity,
                    const NeuronIndexPtr& neuron_index,
                    const TensorPtr& labels) {
  CHECK(x->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(b->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->ndim() == 2, "Weight matrix must be 2D.");
  CHECK(b->ndim() == 1, "Bias must be 1D.");
  CHECK(x->shape().last() == w->shape().last(), "Cols of x and w must match.");
  CHECK(w->shape(0) == b->shape(0), "Rows of w and b must match.");
  CHECK(x->ndim() == 2, "Sparse linear is only supported for 2d inputs.");

  size_t dim = w->shape(0);
  size_t input_dim = w->shape(1);
  size_t nonzeros = sparsity * dim;
  const float* w_ptr = w->data<float>();
  const float* b_ptr = b->data<float>();
  const float* x_ptr = x->data<float>();

  size_t batch_size = x->shape(0);

  auto y_indices = DenseTensor::make(Shape(batch_size * nonzeros), Dtype::u32);
  auto y_values = DenseTensor::make(Shape(batch_size * nonzeros), Dtype::f32);
  auto y_offsets = DenseTensor::make(Shape(batch_size + 1), Dtype::u32);
  Shape y_shape(batch_size, dim);

  const uint32_t* labels_ptr = nullptr;
  const uint32_t* label_offsets_ptr = nullptr;

  if (labels) {
    if (labels->isSparse()) {
      CHECK(labels->shape() == y_shape, "Label shape must match output shape.");
      const auto& csr_labels = csr(labels);
      labels_ptr = csr_labels->colIndices()->data<uint32_t>();
      label_offsets_ptr = csr_labels->rowOffsets()->data<uint32_t>();
    } else {
      CHECK(labels->ndim() == 1,
            "Labels must be dense 1d with dtype u32 or Csr.");
      CHECK(labels->dtype() == Dtype::u32,
            "Labels must be dense 1d with dtype u32 or Csr.");
      CHECK(labels->shape(0) == batch_size,
            "Label shape must match output shape.");

      labels_ptr = dense(labels)->data<uint32_t>();
    }
  }

  uint32_t* y_indices_ptr = y_indices->data<uint32_t>();
  float* y_values_ptr = y_indices->data<float>();
  uint32_t* y_offsets_ptr = y_indices->data<uint32_t>();
  y_offsets_ptr[0] = 0;

  for (size_t n = 0; n < batch_size; n++) {
    if (label_offsets_ptr && labels_ptr) {
      size_t label_start = label_offsets_ptr[n];
      neuron_index->query(
          /*query=*/x_ptr + n * input_dim,
          /*candidates=*/y_indices_ptr + n * nonzeros,
          /*n_candidates=*/nonzeros, /*force_select*/ labels_ptr + label_start,
          /*n_force_select=*/label_offsets_ptr[n + 1] - label_start);
    } else if (labels_ptr) {
      neuron_index->query(
          /*query=*/x_ptr + n * input_dim,
          /*candidates=*/y_indices_ptr + n * nonzeros,
          /*n_candidates=*/nonzeros, /*force_select*/ labels_ptr + n,
          /*n_force_select=*/1);
    }

    sgemsv(/*x=*/x_ptr + n * input_dim, /*w=*/w_ptr, /*b=*/b_ptr,
           /*input_dim=*/input_dim, /*y_indices=*/y_indices_ptr + n * nonzeros,
           /*y_values=*/y_values_ptr + n * nonzeros, /*y_nonzeros=*/nonzeros);
  }

  return CsrTensor::make(y_offsets, y_indices, y_values, y_shape);
}

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const CsrTensorPtr& y_grad) {
  const uint32_t* y_indices_ptr = y_grad->colIndices()->data<uint32_t>();
  const float* y_grad_ptr = y_grad->colValues()->data<float>();
  const uint32_t* y_offsets_ptr = y_grad->rowOffsets()->data<uint32_t>();

  size_t input_dim = w->shape(1);
  size_t batch_size = x->shape(0);

  const float* w_ptr = w->data<float>();
  const float* x_ptr = x->data<float>();

  auto w_grad = DenseTensor::make(w->shape(), w->dtype());
  auto b_grad = DenseTensor::make(b->shape(), b->dtype());
  auto x_grad = DenseTensor::make(x->shape(), x->dtype());

  float* w_grad_ptr = w_grad->data<float>();
  float* b_grad_ptr = b_grad->data<float>();
  float* x_grad_ptr = x_grad->data<float>();

  for (size_t n = 0; n < batch_size; n++) {
    size_t y_offset = y_offsets_ptr[n];

    sgemsvGrad(/*x=*/x_ptr + n * input_dim,
               /*x_grad=*/x_grad_ptr + n * input_dim, w_ptr,
               /*w_grad=*/w_grad_ptr, /*b_grad=*/b_grad_ptr,
               /*input_dim=*/input_dim, /* y_indices=*/y_indices_ptr + y_offset,
               /*y_grad=*/y_grad_ptr + y_offset,
               /*y_nonzeros=*/y_offsets_ptr[n + 1] - y_offset);
  }

  return {x_grad, w_grad, b_grad};
}

}  // namespace thirdai::smx