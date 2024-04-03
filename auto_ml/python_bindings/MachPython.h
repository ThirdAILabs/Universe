#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::automl::mach::python {

void defineMach(py::module_& module);

}  // namespace thirdai::automl::mach::python
