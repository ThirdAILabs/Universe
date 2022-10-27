#pragma once

#include <pybind11/pybind11.h>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createNewDatasetSubmodule(py::module_& module);

}  // namespace thirdai::dataset::python