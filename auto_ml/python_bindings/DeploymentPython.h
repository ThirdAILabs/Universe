#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <auto_ml/src/models/UniversalDeepTransformer.h>
#include <dataset/src/DataSource.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <optional>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::automl::deployment::python {

void createDeploymentSubmodule(py::module_& thirdai_module);

template <typename T>
void defConstantParameter(py::module_& submodule, bool add_docs);

template <typename T>
void defOptionMappedParameter(py::module_& submodule, bool add_docs);

py::object makeUserSpecifiedParameter(const std::string& name,
                                      const py::object& type);

}  // namespace thirdai::automl::deployment::python