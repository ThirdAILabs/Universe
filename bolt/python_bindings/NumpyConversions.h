#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

py::object tensorToNumpy(const tensor::TensorPtr& tensor,
                         bool single_row_to_vector = true);

py::object tensorToNumpyTopK(const tensor::TensorPtr& tensor,
                             bool single_row_to_vector = true,
                             uint32_t top_k = 5);

static void can_convert_tensor_to_numpy(const tensor::TensorPtr& tensor);
}  // namespace thirdai::bolt::nn::python