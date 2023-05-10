#include "NumpyConversions.h"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace thirdai::bolt::nn::python {

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> createArrayCopy(
    const std::vector<T>& data, tensor::Dims shape, bool single_row_to_vector) {
  T* data_copy = new T[data.size()];
  std::copy(data.begin(), data.end(), data_copy);

  py::capsule free_when_done(data_copy,
                             [](void* ptr) { delete static_cast<T*>(ptr); });

  // This is so that if we have a tensor with a single row we return a (N,)
  // numpy array instead of a (N,1). This is useful during inference on a single
  // sample.
  if (shape.size() == 2 && shape.at(0) == 1 && single_row_to_vector) {
    return py::array_t<T, py::array::c_style | py::array::forcecast>(
        shape.back(), data_copy, free_when_done);
  }
  return py::array_t<T, py::array::c_style | py::array::forcecast>(
      shape, data_copy, free_when_done);
}

py::object tensorToNumpy(const tensor::TensorPtr& tensor,
                         bool single_row_to_vector) {
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }

  if (!tensor->valuesPtr()) {
    throw std::runtime_error("Cannot convert ragged tensor to numpy.");
  }

  tensor::Dims shape = tensor->dims();
  shape.back() = *nonzeros;

  auto values = createArrayCopy(
      /* data= */ tensor->values(), /* shape= */ shape,
      /* single_row_to_vector= */ single_row_to_vector);

  if (tensor->indicesPtr()) {
    auto indices = createArrayCopy(
        /* data= */ tensor->indices(), /* shape= */ shape,
        /* single_row_to_vector= */ single_row_to_vector);

    return std::move(py::make_tuple(std::move(indices), std::move(values)));
  }

  return std::move(values);
}

}  // namespace thirdai::bolt::nn::python