#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <dataset/src/DataLoader.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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

ModelPipeline createPipeline(const DeploymentConfigPtr& config,
                             const py::dict& parameters);

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters);

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::PredictConfig>& predict_config);

py::object evaluateOnFileWrapper(
    ModelPipeline& model, const std::string& filename,
    std::optional<bolt::PredictConfig>& predict_config);

py::object predictWrapper(ModelPipeline& model, const std::string& sample,
                          bool use_sparse_inference);

py::object predictTokensWrapper(ModelPipeline& model,
                                const std::vector<uint32_t>& tokens,
                                bool use_sparse_inference);

py::object predictBatchWrapper(ModelPipeline& model,
                               const std::vector<std::string>& samples,
                               bool use_sparse_inference);

py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

py::object convertBoltBatchToNumpy(const BoltBatch& batch);

}  // namespace thirdai::automl::deployment::python