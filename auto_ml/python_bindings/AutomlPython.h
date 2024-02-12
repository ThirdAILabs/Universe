#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::automl::python {

void defineAutomlInModule(py::module_& module);

void createUDTTypesSubmodule(py::module_& module);

void createUDTTemporalSubmodule(py::module_& module);

void createDeploymentSubmodule(py::module_& module);

config::ArgumentMap createArgumentMap(const py::dict& input_args);

}  // namespace thirdai::automl::python