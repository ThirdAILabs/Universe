#pragma once

#include <pybind11/pybind11.h>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createFeaturizationSubmodule(py::module_& dataset_submodule);

}  // namespace thirdai::dataset::python