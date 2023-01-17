#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::licensing::python {

void createLicensingSubmodule(py::module_& module);

}  // namespace thirdai::licensing::python
