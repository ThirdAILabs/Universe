#pragma once

#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::automl::deployment_config::python {

void createDeploymentConfigSubmodule(py::module_& thirdai_module);

template <typename T>
HyperParameterPtr<T> makeConstantParamter(T value);

template <typename T>
void defConstantParameter(py::module_& submodule);

template <typename T>
HyperParameterPtr<T> makeOptionParamter(
    std::unordered_map<std::string, T> values);

template <typename T>
void defOptionParameter(py::module_& submodule);

}  // namespace thirdai::automl::deployment_config::python