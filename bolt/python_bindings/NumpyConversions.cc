#include "NumpyConversions.h"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace thirdai::bolt::nn::python {

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

py::object tensorToNumpy(const tensor::TensorPtr& tensor,
                         bool single_row_to_vector) {
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }

  if (!tensor->activationsPtr()) {
    throw std::runtime_error("Cannot convert ragged tensor to numpy.");
  }

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

}  // namespace thirdai::bolt::nn::python