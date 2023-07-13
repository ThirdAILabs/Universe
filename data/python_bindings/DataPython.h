#pragma once

#include <pybind11/pybind11.h>

namespace thirdai::data::python {

namespace py = pybind11;

void createDataSubmodule(py::module_& dataset_submodule);

}  // namespace thirdai::data::python