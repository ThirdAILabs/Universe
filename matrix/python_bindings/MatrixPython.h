#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace thirdai::matrix::python {

void createMatrixSubmodule(py::module_& module);

}  // namespace thirdai::matrix::python