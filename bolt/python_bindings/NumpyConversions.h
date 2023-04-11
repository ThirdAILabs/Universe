#pragma once

#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

py::object tensorToNumpy(const tensor::TensorPtr& tensor);

}  // namespace thirdai::bolt::nn::python