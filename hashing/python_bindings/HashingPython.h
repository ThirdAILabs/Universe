#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::hashing::python {

void createHashingSubmodule(py::module_& module);

}  // namespace thirdai::hashing::python
