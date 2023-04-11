#include "NumpyConversions.h"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace thirdai::bolt::nn::python {

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> createArray(
    const T* data, uint32_t rows, uint32_t cols) {
  uint64_t flattened_dim = rows * cols;

  T* data_copy = new T[flattened_dim];
  std::copy(data, data + flattened_dim, data_copy);

  py::capsule free_when_done(data_copy,
                             [](void* ptr) { delete static_cast<T*>(ptr); });

  if (rows == 1) {
    return py::array_t<T, py::array::c_style | py::array::forcecast>(
        cols, data_copy, free_when_done);
  }
  return py::array_t<T, py::array::c_style | py::array::forcecast>(
      {rows, cols}, data_copy, free_when_done);
}

py::object tensorToNumpy(const tensor::TensorPtr& tensor) {
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }

  auto activations =
      createArray(tensor->activationsPtr(), tensor->batchSize(), *nonzeros);

  if (tensor->activeNeuronsPtr()) {
    auto active_neurons =
        createArray(tensor->activationsPtr(), tensor->batchSize(), *nonzeros);

    return std::move(
        py::make_tuple(std::move(active_neurons), std::move(activations)));
  }

  return std::move(activations);
}

}  // namespace thirdai::bolt::nn::python