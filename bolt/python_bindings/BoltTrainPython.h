#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

void createBoltTrainSubmodule(py::module_& module);

}  // namespace thirdai::bolt::train::python