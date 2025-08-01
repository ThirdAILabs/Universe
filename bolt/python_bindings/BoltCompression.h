#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::compression::python {

void createCompressionSubmodule(py::module_& module);

}  // namespace thirdai::bolt::compression::python