#include <bolt/src/utils/Timer.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <utils/Logging.h>

namespace thirdai::smx {

inline void sgemsv(const float* x, const float* w, const float* b,
                   size_t input_dim, const uint32_t* y_indices, float* y_values,
                   size_t y_nonzeros) {
  for (size_t n = 0; n < y_nonzeros; n++) {
    const float* w_n = w + y_indices[n] * input_dim;

    float act = b ? b[y_indices[n]] : 0.0;
    for (size_t i = 0; i < input_dim; i++) {
      act += w_n[i] * x[i];
    }
    y_values[n] = act;
  }
}

// inline void sgemsvGrad(const float* x, float* x_grad, const float* w,
//                        float* w_grad, float* b_grad, size_t input_dim,
//                        const uint32_t* y_indices, const float* y_grad,
//                        size_t y_nonzeros) {
//   std::fill(x_grad, x_grad + input_dim, 0);

//   for (size_t n = 0; n < y_nonzeros; n++) {
//     size_t neuron = y_indices[n];
//     const float* w_n = w + neuron * input_dim;
//     float* w_g_n = w_grad + neuron * input_dim;

//     for (size_t i = 0; i < input_dim; i++) {
//       w_g_n[i] += y_grad[n] * x[i];
//       x_grad[i] += y_grad[n] * w_n[i];
//     }

//     if (b_grad) {
//       b_grad[neuron] += y_grad[n];
//     }
//   }
// }

inline void xGrad(float* x_grad, const float* w, size_t input_dim,
                  const uint32_t* y_indices, const float* y_grad,
                  size_t y_nonzeros) {
  std::fill(x_grad, x_grad + input_dim, 0);

  for (size_t n = 0; n < y_nonzeros; n++) {
    size_t neuron = y_indices[n];
    const float* w_n = w + neuron * input_dim;

#pragma omp simd
    for (size_t i = 0; i < input_dim; i++) {
      x_grad[i] += y_grad[n] * w_n[i];
    }
  }
}

CsrTensorPtr linear(const DenseTensorPtr& x, const DenseTensorPtr& w,
                    const DenseTensorPtr& b, float sparsity,
                    const NeuronIndexPtr& neuron_index,
                    const TensorPtr& labels) {
  CHECK(x->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
  CHECK(w->ndim() == 2, "Weight matrix must be 2D.");
  CHECK(x->shape().last() == w->shape().last(), "Cols of x and w must match.");
  CHECK(x->ndim() == 2, "Sparse linear is only supported for 2d inputs.");

  size_t dim = w->shape(0);
  size_t input_dim = w->shape(1);
  size_t nonzeros = sparsity * dim;

  const float* w_ptr = w->data<float>();
  const float* x_ptr = x->data<float>();

  const float* b_ptr = nullptr;
  if (b) {
    CHECK(b->dtype() == Dtype::f32, "Linear only supports f32 tensors.");
    CHECK(b->ndim() == 1, "Bias must be 1D.");
    CHECK(w->shape(0) == b->shape(0), "Rows of w and b must match.");

    b_ptr = b->data<float>();
  }

  size_t batch_size = x->shape(0);

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

  auto y_offsets = DenseTensor::make(Shape(batch_size + 1), Dtype::u32);
  auto y_indices = DenseTensor::make(Shape(batch_size * nonzeros), Dtype::u32);
  auto y_values = DenseTensor::make(Shape(batch_size * nonzeros), Dtype::f32);

  uint32_t* y_offsets_ptr = y_offsets->data<uint32_t>();
  uint32_t* y_indices_ptr = y_indices->data<uint32_t>();
  float* y_values_ptr = y_values->data<float>();
  y_offsets_ptr[0] = 0;

#pragma omp parallel for default(none)                                     \
    shared(batch_size, input_dim, nonzeros, label_offsets_ptr, labels_ptr, \
           neuron_index, x_ptr, w_ptr, b_ptr, y_indices_ptr, y_values_ptr, \
           y_offsets_ptr)
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
    } else {
      neuron_index->query(
          /*query=*/x_ptr + n * input_dim,
          /*candidates=*/y_indices_ptr + n * nonzeros,
          /*n_candidates=*/nonzeros, /*force_select*/ nullptr,
          /*n_force_select=*/0);
    }

    sgemsv(/*x=*/x_ptr + n * input_dim, /*w=*/w_ptr, /*b=*/b_ptr,
           /*input_dim=*/input_dim, /*y_indices=*/y_indices_ptr + n * nonzeros,
           /*y_values=*/y_values_ptr + n * nonzeros, /*y_nonzeros=*/nonzeros);

    y_offsets_ptr[n + 1] = (n + 1) * nonzeros;
  }

  return CsrTensor::make(y_offsets, y_indices, y_values, y_shape);
}

std::tuple<DenseTensorPtr, DenseTensorPtr, DenseTensorPtr> linearGrad(
    const DenseTensorPtr& x, const DenseTensorPtr& w, const DenseTensorPtr& b,
    const CsrTensorPtr& y_grad) {
  const uint32_t* y_offsets_ptr = y_grad->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices_ptr = y_grad->colIndices()->data<uint32_t>();
  const float* y_grad_ptr = y_grad->colValues()->data<float>();

  size_t dim = w->shape(0);
  size_t input_dim = w->shape(1);
  size_t batch_size = x->shape(0);

  const float* w_ptr = w->data<float>();
  const float* x_ptr = x->data<float>();

  bolt::utils::Timer alloc_timer;

  auto w_grad = zeros(w->shape());

  alloc_timer.stop();
  logging::info(fmt::format("smx embedding grad alloc | time {} ms",
                            alloc_timer.milliseconds()));

  auto x_grad = DenseTensor::make(x->shape(), x->dtype());

  float* w_grad_ptr = w_grad->data<float>();
  float* x_grad_ptr = x_grad->data<float>();

  DenseTensorPtr b_grad = nullptr;
  float* b_grad_ptr = nullptr;
  if (b) {
    b_grad = zeros(b->shape());
    b_grad_ptr = b_grad->data<float>();
  }

  bolt::utils::Timer xgrad_timer;

#pragma omp parallel for default(none)                                      \
    shared(batch_size, input_dim, y_offsets_ptr, y_indices_ptr, y_grad_ptr, \
           w_ptr, x_grad_ptr)
  for (size_t n = 0; n < batch_size; n++) {
    const size_t y_offset = y_offsets_ptr[n];
    xGrad(/*x_grad=*/x_grad_ptr + n * input_dim, /*w=*/w_ptr,
          /*input_dim=*/input_dim, /*y_indices=*/y_indices_ptr + y_offset,
          /*y_grad=*/y_grad_ptr + y_offset,
          /*y_nonzeros=*/y_offsets_ptr[n + 1] - y_offset);
  }

  xgrad_timer.stop();
  logging::info(fmt::format("smx linear backward xgrad loop | time {} ms",
                            xgrad_timer.milliseconds()));

  size_t shard_size = std::max(dim / 384, 1UL);

  bolt::utils::Timer wgrad_timer;

#pragma omp parallel for default(none)                            \
    shared(batch_size, dim, input_dim, shard_size, y_offsets_ptr, \
           y_indices_ptr, y_grad_ptr, w_grad_ptr, b_grad_ptr, x_ptr)
  for (size_t start = 0; start < dim; start += shard_size) {
    const size_t end = start + shard_size;
    for (size_t n = 0; n < batch_size; n++) {
      const size_t y_start = y_offsets_ptr[n], y_end = y_offsets_ptr[n + 1];
      for (size_t i = y_start; i < y_end; i++) {
        const size_t neuron = y_indices_ptr[i];
        if (start <= neuron && neuron < end) {
          const float neuron_grad = y_grad_ptr[i];

          float* neuron_w_grad = w_grad_ptr + neuron * input_dim;
          const float* x_n = x_ptr + n * input_dim;

#pragma omp simd
          for (size_t j = 0; j < input_dim; j++) {
            neuron_w_grad[j] += neuron_grad * x_n[j];
          }

          if (b_grad_ptr) {
            b_grad_ptr[neuron] += neuron_grad;
          }
        }
      }
    }
  }

  wgrad_timer.stop();
  logging::info(fmt::format("smx linear backward wgrad loop | time {} ms",
                            wgrad_timer.milliseconds()));

  // #pragma omp parallel for default(none)
  //     shared(batch_size, input_dim, x_ptr, w_ptr, x_grad_ptr, w_grad_ptr,
  //            b_grad_ptr, y_indices_ptr, y_grad_ptr, y_offsets_ptr)
  //   for (size_t n = 0; n < batch_size; n++) {
  //     size_t y_offset = y_offsets_ptr[n];

  //     sgemsvGrad(/*x=*/x_ptr + n * input_dim,
  //                /*x_grad=*/x_grad_ptr + n * input_dim, w_ptr,
  //                /*w_grad=*/w_grad_ptr, /*b_grad=*/b_grad_ptr,
  //                /*input_dim=*/input_dim, /* y_indices=*/y_indices_ptr +
  //                y_offset,
  //                /*y_grad=*/y_grad_ptr + y_offset,
  //                /*y_nonzeros=*/y_offsets_ptr[n + 1] - y_offset);
  //   }

  return {x_grad, w_grad, b_grad};
}

}  // namespace thirdai::smx