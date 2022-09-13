#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::utils::python {

void createUtilsSubmodule(py::module_& module_);

}
