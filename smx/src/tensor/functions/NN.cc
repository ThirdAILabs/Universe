#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <algorithm>

namespace thirdai::smx {

inline void embeddingGather(const uint32_t* indices, const float* values,
                            size_t n_indices, const float* embs,
                            const float* bias, size_t emb_dim, float* out) {
  if (bias) {
    std::copy(bias, bias + emb_dim, out);
  } else {
    std::fill(out, out + emb_dim, 0.0);
  }
  for (size_t i = 0; i < n_indices; i++) {
    const float* emb = embs + indices[i] * emb_dim;
#pragma omp simd
    for (size_t j = 0; j < emb_dim; j++) {
      out[j] += values[i] * emb[j];
    }
  }
}

inline void embeddingGradScatter(const uint32_t* indices, const float* values,
                                 size_t n_indices, float* emb_grad,
                                 size_t emb_dim, const float* out_grad) {
  for (size_t i = 0; i < n_indices; i++) {
    float* start = emb_grad + indices[i] * emb_dim;
#pragma omp simd
    for (size_t j = 0; j < emb_dim; j++) {
      start[j] += values[i] * out_grad[j];
    }
  }
}

DenseTensorPtr embedding(const CsrTensorPtr& indices,
                         const DenseTensorPtr& embeddings,
                         const DenseTensorPtr& bias) {
  CHECK(embeddings->dtype() == Dtype::f32, "Embeddings should have dtype f32.");
  CHECK(indices->dtype() == Dtype::f32, "Index weights should have dtype f32.");
  CHECK(embeddings->ndim() == 2, "Embeddings block should be 2d.");
  CHECK(indices->nDenseCols() == embeddings->shape(0),
        "Indices range should match n-embeddings.");

  const uint32_t* row_offsets = indices->rowOffsets()->data<uint32_t>();
  const uint32_t* col_indices = indices->colIndices()->data<uint32_t>();
  const float* col_values = indices->colValues()->data<float>();
  const float* embeddings_ptr = embeddings->data<float>();

  const float* bias_ptr = nullptr;
  if (bias) {
    CHECK(bias->ndim() == 1, "Bais must be 1D.");
    CHECK(bias->shape(0) == embeddings->shape(1),
          "Bais dim must match embedding dim.");
    bias_ptr = bias->data<float>();
  }

  size_t batch_size = indices->nRows();
  size_t emb_dim = embeddings->shape(1);

  auto out = DenseTensor::make(Shape(batch_size, emb_dim), Dtype::f32);
  float* out_ptr = out->data<float>();

// TODO(Nicholas): change parallelism condition to include number of tokens
// embeddings are being computed for.
#pragma omp parallel for default(none)                                       \
    shared(batch_size, row_offsets, col_indices, col_values, embeddings_ptr, \
           bias_ptr, emb_dim, out_ptr) if (batch_size > 1)
  for (size_t i = 0; i < batch_size; i++) {
    size_t offset = row_offsets[i];
    embeddingGather(/*indices=*/col_indices + offset,
                    /*values=*/col_values + offset, row_offsets[i + 1] - offset,
                    /*embs=*/embeddings_ptr, /*bias=*/bias_ptr,
                    /*emb_dim=*/emb_dim, /*out=*/out_ptr + i * emb_dim);
  }

  return out;
}

std::pair<DenseTensorPtr, DenseTensorPtr> embeddingGrad(
    const CsrTensorPtr& indices, const DenseTensorPtr& out_grad, bool bias) {
  const uint32_t* row_offsets = indices->rowOffsets()->data<uint32_t>();
  const uint32_t* col_indices = indices->colIndices()->data<uint32_t>();
  const float* col_values = indices->colValues()->data<float>();
  const float* out_grad_ptr = out_grad->data<float>();

  size_t batch_size = indices->nRows();
  size_t emb_dim = out_grad->shape(1);

  auto emb_grad = zeros(Shape(indices->shape(1), emb_dim));
  float* emb_grad_ptr = emb_grad->data<float>();

#pragma omp parallel for default(none)                                     \
    shared(batch_size, row_offsets, col_indices, col_values, emb_grad_ptr, \
           emb_dim, out_grad_ptr)
  for (size_t i = 0; i < batch_size; i++) {
    size_t offset = row_offsets[i];
    embeddingGradScatter(col_indices + offset, col_values + offset,
                         row_offsets[i + 1] - offset, emb_grad_ptr, emb_dim,
                         out_grad_ptr + i * emb_dim);
  }

  DenseTensorPtr bias_grad = nullptr;
  if (bias) {
    bias_grad = DenseTensor::make(Shape(emb_dim), Dtype::f32);

    bias_grad->eigenMatrix<float>() =
        out_grad->eigenMatrix<float>().colwise().sum();
  }

  // TODO(Nicholas) should we add option for input grads?

  return {emb_grad, bias_grad};
}

}  // namespace thirdai::smx