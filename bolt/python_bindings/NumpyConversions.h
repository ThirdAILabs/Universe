#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

py::object tensorToNumpy(const TensorPtr& tensor,
                         bool single_row_to_vector = true);

py::object tensorToNumpyTopK(const TensorPtr& tensor,
                             bool single_row_to_vector = true,
                             uint32_t top_k = 5);

static void can_convert_tensor_to_numpy(const TensorPtr& tensor);

}  // namespace thirdai::bolt::python