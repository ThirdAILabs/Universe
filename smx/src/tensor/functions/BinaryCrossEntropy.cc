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
         " in binary_cross_entropy with output dim " + std::to_string(dim) +
         ".";
}

inline void checkLabel(uint32_t label, size_t dim) {
  if (label >= dim) {
    throw std::invalid_argument(invalidLabelError(label, dim));
  }
}

std::pair<DenseTensorPtr, DenseTensorPtr> sparseBinaryCrossEntropy(
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

  auto y = dense(sigmoid(logits));

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_ptr = labels->data<uint32_t>();

  float loss = 0.0;

  for (size_t i = 0; i < n_rows; i++) {
    const float* activations = y_ptr + i * dim;
    for (size_t j = 0; j < dim; j++) {
      loss -= std::log(1 - activations[j]);
    }

    const uint32_t label = label_ptr[i];
    checkLabel(label, dim);

    loss -= (std::log(activations[label]) - std::log(1 - activations[label]));
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, DenseTensorPtr> sparseBinaryCrossEntropy(
    const DenseTensorPtr& logits, const CsrTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::f32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  CHECK(logits->shape() == labels->shape(),
        "Labels should shape should match logits shape.");

  auto y = dense(sigmoid(logits));

  const size_t dim = y->shape().last();
  const size_t n_rows = y->size() / dim;

  const float* y_ptr = y->data<float>();
  const uint32_t* label_offsets = labels->rowOffsets()->data<uint32_t>();
  const uint32_t* label_indices = labels->colIndices()->data<uint32_t>();
  const float* label_values = labels->colValues()->data<float>();

  float loss = 0.0;
  for (size_t i = 0; i < n_rows; i++) {
    const float* activations = y_ptr + i * dim;
    for (size_t j = 0; j < dim; j++) {
      loss -= std::log(1 - activations[j]);
    }

    const size_t start = label_offsets[i], end = label_offsets[i + 1];
    for (size_t j = start; j < end; j++) {
      const uint32_t label = label_indices[j];
      checkLabel(label, dim);

      const float label_act = activations[label];
      const float label_val = label_values[j];

      loss += std::log(1 - label_act);

      loss -= label_val * std::log(label_act) +
              (1 - label_val) * std::log(1 - label_act);
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, CsrTensorPtr> sparseBinaryCrossEntropy(
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

  auto y = csr(sigmoid(logits));

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
      } else {
        loss -= std::log(1 - y_values[j]);
      }
    }
    if (!found) {
      loss -= std::log(1e-7);  // Use small activation if label not found.
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

std::pair<DenseTensorPtr, CsrTensorPtr> sparseBinaryCrossEntropy(
    const CsrTensorPtr& logits, const CsrTensorPtr& labels) {
  CHECK(labels->dtype() == Dtype::f32, "Labels should have dtype u32.");
  CHECK(logits->dtype() == Dtype::f32, "Outputs should have dtype f32.");

  CHECK(logits->shape() == labels->shape(),
        "Labels should shape should match logits shape.");

  auto y = csr(sigmoid(logits));

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

    for (size_t j = y_start; j < y_end; j++) {
      loss -= std::log(1 - y_values[j]);
    }

    for (size_t j = label_start; j < label_end; j++) {
      const uint32_t label = label_indices[j];
      checkLabel(label, dim);
      const uint32_t* loc =
          std::find(y_indices + y_start, y_indices + y_end, label);

      const float label_val = label_values[j];

      if (loc != y_indices + y_end) {
        const float label_act = y_values[loc - y_indices];
        loss += std::log(1 - label_act);

        loss -= label_val * std::log(label_act) +
                (1 - label_val) * std::log(1 - label_act);
      } else {
        loss -=
            label_val * std::log(1e-7) + (1 - label_val) * std::log(1 - 1e-7);
      }
    }
  }
  loss /= n_rows;

  return {scalar(loss), y};
}

DenseTensorPtr sparseBinaryCrossEntropyGrad(const DenseTensorPtr& y,
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
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return logits_grad;
}

DenseTensorPtr sparseBinaryCrossEntropyGrad(const DenseTensorPtr& y,
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

    for (size_t j = 0; j < dim; j++) {
      logits_grad_ptr[i * dim + j] = -y_ptr[i * dim + j] / n_rows;
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

CsrTensorPtr sparseBinaryCrossEntropyGrad(const CsrTensorPtr& y,
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

CsrTensorPtr sparseBinaryCrossEntropyGrad(const CsrTensorPtr& y,
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

    for (size_t j = y_start; j < y_end; j++) {
      const uint32_t* loc = std::find(label_indices + label_start,
                                      label_indices + label_end, y_indices[j]);

      float label = loc != label_indices + label_end
                        ? label_values[loc - label_indices]
                        : 0.0;
      logits_grad_ptr[j] = (label - y_values[j]) / n_rows;
    }
  }

  if (invalid_label) {
    throw std::invalid_argument(invalidLabelError(*invalid_label, dim));
  }

  return CsrTensor::make(y->rowOffsets(), y->colIndices(), logits_grad,
                         y->shape());
}

std::pair<DenseTensorPtr, TensorPtr> sparseBinaryCrossEntropy(
    const TensorPtr& logits, const TensorPtr& labels) {
  if (logits->isSparse()) {
    if (labels->isSparse()) {
      return sparseBinaryCrossEntropy(csr(logits), csr(labels));
    }
    return sparseBinaryCrossEntropy(csr(logits), dense(labels));
  }
  if (labels->isSparse()) {
    return sparseBinaryCrossEntropy(dense(logits), csr(labels));
  }
  return sparseBinaryCrossEntropy(dense(logits), dense(labels));
}

TensorPtr sparseBinaryCrossEntropyGrad(const TensorPtr& y,
                                       const TensorPtr& labels) {
  if (y->isSparse()) {
    if (labels->isSparse()) {
      return sparseBinaryCrossEntropyGrad(csr(y), csr(labels));
    }
    return sparseBinaryCrossEntropyGrad(csr(y), dense(labels));
  }
  if (labels->isSparse()) {
    return sparseBinaryCrossEntropyGrad(dense(y), csr(labels));
  }
  return sparseBinaryCrossEntropyGrad(dense(y), dense(labels));
}

}  // namespace thirdai::smx