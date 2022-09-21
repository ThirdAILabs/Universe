#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void defineAutoClassifeirsInModule(py::module_& bolt_submodule);

}  // namespace thirdai::bolt::python
