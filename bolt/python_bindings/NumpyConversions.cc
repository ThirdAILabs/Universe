#include "NumpyConversions.h"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace thirdai::bolt::python {

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> createArrayCopy(
    const T* data, uint32_t rows, uint32_t cols, bool single_row_to_vector) {
  uint64_t flattened_dim = rows * cols;

  T* data_copy = new T[flattened_dim];
  std::copy(data, data + flattened_dim, data_copy);

  py::capsule free_when_done(data_copy,
                             [](void* ptr) { delete static_cast<T*>(ptr); });

  // This is so that if we have a tensor with a single row we return a (N,)
  // numpy array instead of a (N,1). This is useful during inference on a single
  // sample.
  if (rows == 1 && single_row_to_vector) {
    return py::array_t<T, py::array::c_style | py::array::forcecast>(
        cols, data_copy, free_when_done);
  }
  return py::array_t<T, py::array::c_style | py::array::forcecast>(
      {rows, cols}, data_copy, free_when_done);
}

void can_convert_tensor_to_numpy(const TensorPtr& tensor) {
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }

  if (!tensor->activationsPtr()) {
    throw std::runtime_error("Cannot convert ragged tensor to numpy.");
  }
}

py::object tensorToNumpy(const TensorPtr& tensor, bool single_row_to_vector) {
  auto nonzeros = tensor->nonzeros();

  can_convert_tensor_to_numpy(tensor);
  auto activations =
      createArrayCopy(/* data= */ tensor->activationsPtr(),
                      /* rows= */ tensor->batchSize(), /* cols= */ *nonzeros,
                      /* single_row_to_vector= */ single_row_to_vector);

  if (tensor->activeNeuronsPtr()) {
    auto active_neurons = createArrayCopy(
        /* data= */ tensor->activeNeuronsPtr(), /* rows= */ tensor->batchSize(),
        /* cols= */ *nonzeros,
        /* single_row_to_vector= */ single_row_to_vector);

    return std::move(
        py::make_tuple(std::move(active_neurons), std::move(activations)));
  }

  return std::move(activations);
}

py::object tensorToNumpyTopK(const TensorPtr& tensor, bool single_row_to_vector,
                             uint32_t top_k) {
  can_convert_tensor_to_numpy(tensor);

  std::pair<std::vector<uint32_t>, std::vector<float> > topkIdxValuePair =
      tensor->topKIndexValuePair(top_k);

  const uint32_t* flattened_active_neurons = topkIdxValuePair.first.data();
  const float* flattened_activations = topkIdxValuePair.second.data();

  auto activations = createArrayCopy(flattened_activations, tensor->batchSize(),
                                     top_k, single_row_to_vector);
  auto active_neurons =
      createArrayCopy(flattened_active_neurons, tensor->batchSize(), top_k,
                      single_row_to_vector);
  return std::move(
      py::make_tuple(std::move(active_neurons), std::move(activations)));
}

TensorPtr fromNumpySparse(const NumpyArray<uint32_t>& indices,
                          const NumpyArray<float>& values, size_t last_dim,
                          bool with_grad) {
  if (indices.ndim() != 2) {
    throw std::invalid_argument("Expected indices to be 2D.");
  }
  if (values.ndim() != 2) {
    throw std::invalid_argument("Expected values to be 2D.");
  }

  size_t batch_size = indices.shape(0);
  size_t nonzeros = indices.shape(1);

  return Tensor::fromArray(indices.data(), values.data(), batch_size, last_dim,
                           nonzeros, /* with_grad= */ with_grad);
}

TensorPtr fromNumpyDense(const NumpyArray<float>& values, bool with_grad) {
  if (values.ndim() != 2) {
    throw std::invalid_argument("Expected values to be 2D.");
  }

  size_t batch_size = values.shape(0);
  size_t dim = values.shape(1);

  return Tensor::fromArray(nullptr, values.data(), batch_size, dim,
                           /* nonzeros= */ dim, /* with_grad= */ with_grad);
}

}  // namespace thirdai::bolt::python
