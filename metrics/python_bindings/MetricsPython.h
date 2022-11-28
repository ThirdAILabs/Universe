#pragma once

#include <pybind11/pybind11.h>

namespace thirdai::metrics::python {

namespace py = pybind11;

void createMetricsSubmodule(py::module_& dataset_submodule);

}  // namespace thirdai::metrics::python