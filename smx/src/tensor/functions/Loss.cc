#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::smx {

std::pair<DenseTensorPtr, DenseTensorPtr> sparseCrossEntropy(
    const DenseTensorPtr& logits, const DenseTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::u32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  const auto& logits_shape = logits->shape();
  const auto& labels_shape = labels->shape();

  CHECK(logits_shape.ndim() - 1 == labels_shape.ndim(),
        "Labels should shape should match logits up to last dimension.");
  for (size_t i = 0; i < logits_shape.ndim() - 1; i++) {
    CHECK(logits_shape[i] == labels_shape[i],
          "Labels should shape should match logits up to last dimension.");
  }

  auto y = dense(softmax(logits));

  size_t dim = y->shape().last();
  size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    uint32_t label = label_ptr[i];
    if (label >= dim) {
      throw std::invalid_argument("Invalid label " + std::to_string(label) +
                                  " in cross_entropy with output dim " +
                                  std::to_string(dim) + ".");
    }
    loss -= std::log(y_ptr[i * dim + label]);
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, CsrTensorPtr> sparseCrossEntropy(
    const CsrTensorPtr& logits, const DenseTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::u32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  const auto& logits_shape = logits->shape();
  const auto& labels_shape = labels->shape();

  CHECK(logits_shape.ndim() - 1 == labels_shape.ndim(),
        "Labels should shape should match logits up to last dimension.");
  for (size_t i = 0; i < logits_shape.ndim() - 1; i++) {
    CHECK(logits_shape[i] == labels_shape[i],
          "Labels should shape should match logits up to last dimension.");
  }

  auto y = csr(softmax(logits));

  size_t dim = y->nDenseCols();
  size_t n_rows = y->nRows();

  const uint32_t* y_offsets = y->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices = y->colIndices()->data<uint32_t>();
  const float* y_values = y->colValues()->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    uint32_t label = label_ptr[i];
    if (label >= dim) {
      throw std::invalid_argument("Invalid label " + std::to_string(label) +
                                  " in cross_entropy with output dim " +
                                  std::to_string(dim) + ".");
    }
    bool found = false;
    size_t start = y_offsets[i], end = y_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      if (y_indices[j] == label) {
        loss -= std::log(y_values[j]);
        found = true;
      }
    }
    if (!found) {
      loss -= std::log(1e-7);  // Use small activation if label not found.
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

DenseTensorPtr sparseCrossEntropyGrad(const DenseTensorPtr& y,
                                      const DenseTensorPtr& labels) {
  auto logits_grad = DenseTensor::make(y->shape(), y->dtype());

  size_t dim = y->shape().last();
  size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();
  float* logits_grad_ptr = logits_grad->data<float>();

  std::optional<uint32_t> invalid_label;
#pragma omp parallel for default(none) \
    shared(n_rows, dim, label_ptr, y_ptr, logits_grad_ptr, invalid_label)
  for (size_t i = 0; i < n_rows; i++) {
    uint32_t label = label_ptr[i];
    if (label >= dim) {
#pragma omp critical
      invalid_label = label;
      continue;
    }
    for (size_t j = 0; j < dim; j++) {
      logits_grad_ptr[i * dim + j] =
          ((label == j) - y_ptr[i * dim + j]) / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(
        "Invalid label " + std::to_string(*invalid_label) +
        " in cross_entropy with output dim " + std::to_string(dim) + ".");
  }

  return logits_grad;
}

CsrTensorPtr sparseCrossEntropyGrad(const CsrTensorPtr& y,
                                    const DenseTensorPtr& labels) {
  size_t dim = y->nDenseCols();
  size_t n_rows = y->nRows();

  const uint32_t* y_offsets = y->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices = y->colIndices()->data<uint32_t>();
  const float* y_values = y->colValues()->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  auto logits_grad = DenseTensor::make(y->colIndices()->shape(), Dtype::f32);
  float* logits_grad_ptr = logits_grad->data<float>();

  std::optional<uint32_t> invalid_label;
#pragma omp parallel for default(none)                             \
    shared(n_rows, dim, label_ptr, y_offsets, y_indices, y_values, \
           logits_grad_ptr, invalid_label)
  for (size_t i = 0; i < n_rows; i++) {
    uint32_t label = label_ptr[i];
    if (label >= dim) {
#pragma omp critical
      invalid_label = label;
      continue;
    }
    size_t start = y_offsets[i], end = y_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      logits_grad_ptr[j] = ((label == y_indices[j]) - y_values[j]) / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(
        "Invalid label " + std::to_string(*invalid_label) +
        " in cross_entropy with output dim " + std::to_string(dim) + ".");
  }

  return CsrTensor::make(y->rowOffsets(), y->colIndices(), logits_grad,
                         y->shape());
}

}  // namespace thirdai::smx