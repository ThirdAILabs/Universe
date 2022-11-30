#pragma once

#include <pybind11/pybind11.h>

namespace thirdai::telemetry::python {

namespace py = pybind11;

void createTelemetrySubmodule(py::module_& dataset_submodule);

}  // namespace thirdai::telemetry::python