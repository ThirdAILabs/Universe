#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::mach::python {

void defineMach(py::module_& module);

}  // namespace thirdai::mach::python
