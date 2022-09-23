#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace thirdai::automl::deployment_config::python {

void createDeploymentConfigSubmodule(py::module_& thirdai_module);

py::object createConstantParameter(const py::object& obj);

}  // namespace thirdai::automl::deployment_config::python