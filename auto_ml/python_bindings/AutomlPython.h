#pragma once

#include <auto_ml/src/models/Generator.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <auto_ml/src/models/UniversalDeepTransformer.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::automl::python {

using models::ModelPipeline;
using models::QueryCandidateGenerator;
using models::UniversalDeepTransformer;

void defineAutomlInBoltSubmodule(py::module_& bolt_submodule);

void createModelsSubmodule(py::module_& bolt_submodule);

// Python wrappers for ModelPipline methods

ModelPipeline createPipeline(const deployment::DeploymentConfigPtr& config,
                             const py::dict& parameters);

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters);

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::EvalConfig>& eval_config);

py::object evaluateOnFileWrapper(ModelPipeline& model,
                                 const std::string& filename,
                                 std::optional<bolt::EvalConfig>& eval_config);

template <typename Model, typename InputType>
py::object predictWrapper(Model& model, const InputType& sample,
                          bool use_sparse_inference);

py::object predictTokensWrapper(ModelPipeline& model,
                                const std::vector<uint32_t>& tokens,
                                bool use_sparse_inference);

template <typename Model, typename InputBatchType>
py::object predictBatchWrapper(Model& model, const InputBatchType& samples,
                               bool use_sparse_inference);

// UDT Factory
class UDTFactory {
 public:
  static QueryCandidateGenerator buildUDTGeneratorWrapper(
      py::object& obj, const std::string& source_column,
      const std::string& target_column, const std::string& dataset_size);

  static UniversalDeepTransformer buildUDTClassifierWrapper(
      py::object& obj, data::ColumnDataTypes data_types,
      data::UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col, std::optional<uint32_t> n_target_classes,
      bool integer_target, std::string time_granularity, uint32_t lookahead,
      char delimiter, const std::optional<std::string>& model_config,
      const std::unordered_map<std::string, std::string>& options);

  // These need to be here instead of inside UDTFactory because otherwise I was
  // getting weird linking errors
  static constexpr uint8_t UDT_GENERATOR_IDENTIFIER = 0;
  static constexpr uint8_t UDT_CLASSIFIER_IDENTIFIER = 1;

  static void save_classifier(const UniversalDeepTransformer& classifier,
                              const std::string& filename);

  static void save_generator(const QueryCandidateGenerator& generator,
                             const std::string& filename);

  static py::object load(const std::string& filename);
};

// TODO(Nicholas): Move these to central location and use as helpers here and in
// bolt.
//  Helper functions for numpy conversions.
py::object convertInferenceTrackerToNumpy(bolt::InferenceOutputTracker& output);

py::object convertBoltVectorToNumpy(const BoltVector& vector);

py::object convertBoltBatchToNumpy(const BoltBatch& batch);

}  // namespace thirdai::automl::python