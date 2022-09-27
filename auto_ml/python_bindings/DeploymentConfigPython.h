#pragma once

#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/DataLoader.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::automl::deployment_config::python {

void createDeploymentConfigSubmodule(py::module_& thirdai_module);

template <typename T>
void defConstantParameter(py::module_& submodule);

template <typename T>
void defOptionParameter(py::module_& submodule);

py::object makeUserSpecifiedParameter(const std::string& name,
                                      const py::object& type);

py::object evaluateWrapperDataLoader(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source);

py::object evaluateWrapperFilename(ModelPipeline& model,
                                   const std::string& filename);

py::object predictWrapper(ModelPipeline& model, const std::string& sample);

py::list predictBatchWrapper(ModelPipeline& model,
                             const std::vector<std::string>& samples);

ModelPipeline createPipeline(const DeploymentConfigPtr& config,
                             const std::optional<std::string>& option,
                             const py::dict& parameters);

py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

}  // namespace thirdai::automl::deployment_config::python