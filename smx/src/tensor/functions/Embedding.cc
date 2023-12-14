#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Functions.h>

namespace thirdai::smx {

inline void embeddingGather(const uint32_t* indices, const float* values,
                            size_t n_indices, const float* embs, size_t emb_dim,
                            float* out, bool reduce_mean) {
  for (size_t i = 0; i < n_indices; i++) {
    const float* start = embs + indices[i] * emb_dim;
#pragma omp simd
    for (size_t j = 0; j < emb_dim; j++) {
      out[j] += values[i] * start[j];
    }
  }
  if (reduce_mean) {
#pragma omp simd
    for (size_t i = 0; i < emb_dim; i++) {
      out[i] /= n_indices;
    }
  }
}

inline void embeddingGradScatter(const uint32_t* indices, const float* values,
                                 size_t n_indices, float* emb_grad,
                                 size_t emb_dim, const float* out_grad,
                                 bool reduce_mean) {
  float scale = reduce_mean ? 1.0 / n_indices : 1.0;
  for (size_t i = 0; i < n_indices; i++) {
    float* start = emb_grad + indices[i] * emb_dim;
#pragma omp simd
    for (size_t j = 0; j < emb_dim; j++) {
      start[j] += scale * values[i] * out_grad[j];
    }
  }
}

DenseTensorPtr embedding(const CsrTensorPtr& indices,
                         const DenseTensorPtr& embeddings, bool reduce_mean) {
  CHECK(embeddings->dtype() == Dtype::f32, "Embeddings should have dtype f32.");
  CHECK(indices->dtype() == Dtype::f32, "Index weights should have dtype f32.");
  CHECK(embeddings->ndim() == 2, "Embeddings block should be 2d.");
  CHECK(indices->nDenseCols() == embeddings->shapeAt(0),
        "Indices range should match n-embeddings.");

  const uint32_t* row_offsets = indices->rowOffsets()->data<uint32_t>();
  const uint32_t* col_indices = indices->colIndices()->data<uint32_t>();
  const float* col_values = indices->colValues()->data<float>();
  const float* embeddings_ptr = embeddings->data<float>();

  size_t batch_size = indices->nRows();
  size_t emb_dim = embeddings->shapeAt(1);

  auto out = DenseTensor::make(Shape(batch_size, emb_dim), Dtype::f32);
  float* out_ptr = out->data<float>();

#pragma omp parallel for default(none)                                       \
    shared(batch_size, row_offsets, col_indices, col_values, embeddings_ptr, \
           emb_dim, out_ptr, reduce_mean)
  for (size_t i = 0; i < batch_size; i++) {
    size_t offset = row_offsets[i];
    embeddingGather(col_indices + offset, col_values + offset,
                    row_offsets[i + 1] - offset, embeddings_ptr, emb_dim,
                    out_ptr + i * emb_dim, reduce_mean);
  }

  return out;
}

DenseTensorPtr embeddingGrad(const CsrTensorPtr& indices,
                             const DenseTensorPtr& out_grad, bool reduce_mean) {
  const uint32_t* row_offsets = indices->rowOffsets()->data<uint32_t>();
  const uint32_t* col_indices = indices->colIndices()->data<uint32_t>();
  const float* col_values = indices->colValues()->data<float>();
  const float* out_grad_ptr = out_grad->data<float>();

  size_t batch_size = indices->nRows();
  size_t emb_dim = out_grad->shapeAt(1);

  auto emb_grad =
      DenseTensor::make(Shape(indices->shapeAt(1), emb_dim), Dtype::f32);
  float* emb_grad_ptr = emb_grad->data<float>();

#pragma omp parallel for default(none)                                     \
    shared(batch_size, row_offsets, col_indices, col_values, emb_grad_ptr, \
           emb_dim, out_grad_ptr, reduce_mean)
  for (size_t i = 0; i < batch_size; i++) {
    size_t offset = row_offsets[i];
    embeddingGradScatter(col_indices + offset, col_values + offset,
                         row_offsets[i + 1] - offset, emb_grad_ptr, emb_dim,
                         out_grad_ptr + i * emb_dim, reduce_mean);
  }

  // TODO(Nicholas) should we add option for input grads?

  return emb_grad;
}

}  // namespace thirdai::smx