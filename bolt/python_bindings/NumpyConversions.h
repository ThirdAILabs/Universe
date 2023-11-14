#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object tensorToNumpy(const TensorPtr& tensor,
                         bool single_row_to_vector = true);

py::object tensorToNumpyTopK(const TensorPtr& tensor,
                             bool single_row_to_vector = true,
                             uint32_t top_k = 5);

TensorPtr fromNumpySparse(const NumpyArray<uint32_t>& indices,
                          const NumpyArray<float>& values, size_t last_dim,
                          bool with_grad);

TensorPtr fromNumpyDense(const NumpyArray<float>& values, bool with_grad);

static void can_convert_tensor_to_numpy(const TensorPtr& tensor);

}  // namespace thirdai::bolt::python