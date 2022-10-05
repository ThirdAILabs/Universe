#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/dataset_configs/SingleBlockDatasetFactory.h>
#include <dataset/src/DataLoader.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::automl::deployment::python {

void createDeploymentSubmodule(py::module_& thirdai_module);

template <typename T>
void defConstantParameter(py::module_& submodule);

template <typename T>
void defOptionMappedParameter(py::module_& submodule);

py::object makeUserSpecifiedParameter(const std::string& name,
                                      const py::object& type);

DatasetLoaderFactoryPtr createSingleBlockDatasetFactory(
    const SingleBlockDatasetFactoryConfig& config, const py::dict& parameters);

ModelPipeline createPipeline(const DeploymentConfigPtr& config,
                             const py::dict& parameters);

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters);

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source);

py::object evaluateOnFileWrapper(ModelPipeline& model,
                                 const std::string& filename);

py::object predictWrapper(ModelPipeline& model, const std::string& sample);

py::object predictTokensWrapper(ModelPipeline& model,
                                const std::vector<uint32_t>& tokens);

py::object predictBatchWrapper(ModelPipeline& model,
                               const std::vector<std::string>& samples);

py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

py::object convertBoltBatchToNumpy(const BoltBatch& batch);

}  // namespace thirdai::automl::deployment::python