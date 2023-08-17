#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

void createBoltNNSubmodule(py::module_& module);

}  // namespace thirdai::bolt::nn::python