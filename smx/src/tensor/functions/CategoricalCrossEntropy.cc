#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::smx {

inline std::string invalidLabelError(uint32_t label, size_t dim) {
  return "Invalid label " + std::to_string(label) +
         " in cross_entropy with output dim " + std::to_string(dim) + ".";
}

inline void checkLabel(uint32_t label, size_t dim) {
  if (label >= dim) {
    throw std::invalid_argument(invalidLabelError(label, dim));
  }
}

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

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    const uint32_t label = label_ptr[i];
    checkLabel(label, dim);

    loss -= std::log(y_ptr[i * dim + label]);
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, DenseTensorPtr> sparseCrossEntropy(
    const DenseTensorPtr& logits, const CsrTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::f32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  CHECK(logits->shape() == labels->shape(),
        "Labels should shape should match logits shape.");

  auto y = dense(softmax(logits));

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_offsets = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* label_indices = labels->colIndices()->data<uint32_t>();
  const float* label_values = labels->colValues()->data<float>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    const size_t start = label_offsets[i], end = label_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      const uint32_t label = label_indices[j];
      checkLabel(label, dim);

      loss -= label_values[j] * std::log(y_ptr[i * dim + label]);
    }
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

  const size_t dim = y->nDenseCols();
  const size_t n_rows = y->nRows();

  const uint32_t* y_offsets = y->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices = y->colIndices()->data<uint32_t>();
  const float* y_values = y->colValues()->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    const uint32_t label = label_ptr[i];
    checkLabel(label, dim);

    bool found = false;
    const size_t start = y_offsets[i], end = y_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      if (y_indices[j] == label) {
        loss -= std::log(y_values[j]);
        found = true;
        break;
      }
    }
    if (!found) {
      loss -= std::log(1e-7);  // Use small activation if label not found.
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, CsrTensorPtr> sparseCrossEntropy(
    const CsrTensorPtr& logits, const CsrTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::f32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  CHECK(logits->shape() == labels->shape(),
        "Labels should shape should match logits shape.");

  auto y = csr(softmax(logits));

  const size_t dim = y->nDenseCols();
  const size_t n_rows = y->nRows();

  const uint32_t* y_offsets = y->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices = y->colIndices()->data<uint32_t>();
  const float* y_values = y->colValues()->data<float>();
  const uint32_t* label_offsets = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* label_indices = labels->colIndices()->data<uint32_t>();
  const float* label_values = labels->colValues()->data<float>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    const size_t label_start = label_offsets[i],
                 label_end = label_offsets[i + 1];
    const size_t y_start = y_offsets[i], y_end = y_offsets[i + 1];
    for (size_t j = label_start; j < label_end; j++) {
      const uint32_t label = label_indices[j];
      checkLabel(label, dim);

      const uint32_t* loc =
          std::find(y_indices + y_start, y_indices + y_end, label);
      if (loc != y_indices + y_end) {
        loss -= label_values[j] * std::log(y_values[loc - y_indices]);
      } else {
        loss -= label_values[j] * std::log(1e-7);
      }
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

DenseTensorPtr sparseCrossEntropyGrad(const DenseTensorPtr& y,
                                      const DenseTensorPtr& labels) {
  auto logits_grad = DenseTensor::make(y->shape(), y->dtype());

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();
  float* logits_grad_ptr = logits_grad->data<float>();

  std::optional<uint32_t> invalid_label;
#pragma omp parallel for default(none) \
    shared(n_rows, dim, label_ptr, y_ptr, logits_grad_ptr, invalid_label)
  for (size_t i = 0; i < n_rows; i++) {
    const uint32_t label = label_ptr[i];
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
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return logits_grad;
}

DenseTensorPtr sparseCrossEntropyGrad(const DenseTensorPtr& y,
                                      const CsrTensorPtr& labels) {
  auto logits_grad = DenseTensor::make(y->shape(), y->dtype());

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_offsets = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* label_indices = labels->colIndices()->data<uint32_t>();
  const float* label_values = labels->colValues()->data<float>();
  float* logits_grad_ptr = logits_grad->data<float>();

  std::optional<uint32_t> invalid_label;
#pragma omp parallel for default(none)                                     \
    shared(n_rows, dim, label_offsets, label_indices, label_values, y_ptr, \
           logits_grad_ptr, invalid_label)
  for (size_t i = 0; i < n_rows; i++) {
    const size_t start = label_offsets[i], end = label_offsets[i + 1];

    const float total_label = std::reduce(
        label_values + start, label_values + end, 0.0, std::plus<>());

    for (size_t j = 0; j < dim; j++) {
      logits_grad_ptr[i * dim + j] = -total_label * y_ptr[i * dim + j] / n_rows;
    }

    for (size_t j = start; j < end; j++) {
      const uint32_t label = label_indices[j];
      if (label >= dim) {
#pragma omp critical
        invalid_label = label;
        continue;
      }
      logits_grad_ptr[i * dim + label] += label_values[j] / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return logits_grad;
}

CsrTensorPtr sparseCrossEntropyGrad(const CsrTensorPtr& y,
                                    const DenseTensorPtr& labels) {
  const size_t dim = y->nDenseCols();
  const size_t n_rows = y->nRows();

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
    const uint32_t label = label_ptr[i];
    if (label >= dim) {
#pragma omp critical
      invalid_label = label;
      continue;
    }
    const size_t start = y_offsets[i], end = y_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      logits_grad_ptr[j] = ((label == y_indices[j]) - y_values[j]) / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return CsrTensor::make(y->rowOffsets(), y->colIndices(), logits_grad,
                         y->shape());
}

CsrTensorPtr sparseCrossEntropyGrad(const CsrTensorPtr& y,
                                    const CsrTensorPtr& labels) {
  const size_t dim = y->nDenseCols();
  const size_t n_rows = y->nRows();

  const uint32_t* y_offsets = y->rowOffsets()->data<uint32_t>();
  const uint32_t* y_indices = y->colIndices()->data<uint32_t>();
  const float* y_values = y->colValues()->data<float>();

  const uint32_t* label_offsets = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* label_indices = labels->colIndices()->data<uint32_t>();
  const float* label_values = labels->colValues()->data<float>();

  auto logits_grad = DenseTensor::make(y->colIndices()->shape(), Dtype::f32);
  float* logits_grad_ptr = logits_grad->data<float>();

  std::optional<uint32_t> invalid_label;
#pragma omp parallel for default(none)                                         \
    shared(n_rows, dim, label_offsets, label_indices, label_values, y_offsets, \
           y_indices, y_values, logits_grad_ptr, invalid_label)
  for (size_t i = 0; i < n_rows; i++) {
    const size_t y_start = y_offsets[i], y_end = y_offsets[i + 1];
    const size_t label_start = label_offsets[i],
                 label_end = label_offsets[i + 1];

    const float total_label =
        std::reduce(label_values + label_start, label_values + label_end, 0.0,
                    std::plus<>());

    for (size_t j = y_start; j < y_end; j++) {
      const uint32_t* loc = std::find(label_indices + label_start,
                                      label_indices + label_end, y_indices[j]);

      const float label = loc != label_indices + label_end
                              ? label_values[loc - label_indices]
                              : 0.0;
      logits_grad_ptr[j] = (label - total_label * y_values[j]) / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return CsrTensor::make(y->rowOffsets(), y->colIndices(), logits_grad,
                         y->shape());
}

std::pair<DenseTensorPtr, TensorPtr> sparseCrossEntropy(
    const TensorPtr& logits, const TensorPtr& labels) {
  if (logits->isSparse()) {
    if (labels->isSparse()) {
      return sparseCrossEntropy(csr(logits), csr(labels));
    }
    return sparseCrossEntropy(csr(logits), dense(labels));
  }
  if (labels->isSparse()) {
    return sparseCrossEntropy(dense(logits), csr(labels));
  }
  return sparseCrossEntropy(dense(logits), dense(labels));
}

TensorPtr sparseCrossEntropyGrad(const TensorPtr& y, const TensorPtr& labels) {
  if (y->isSparse()) {
    if (labels->isSparse()) {
      return sparseCrossEntropyGrad(csr(y), csr(labels));
    }
    return sparseCrossEntropyGrad(csr(y), dense(labels));
  }
  if (labels->isSparse()) {
    return sparseCrossEntropyGrad(dense(y), csr(labels));
  }
  return sparseCrossEntropyGrad(dense(y), dense(labels));
}

}  // namespace thirdai::smx