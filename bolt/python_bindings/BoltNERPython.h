#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void addNERModels(py::module_& module);

}  // namespace thirdai::bolt::python