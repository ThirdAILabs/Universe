#pragma once

#include <auto_ml/src/models/Generator.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <auto_ml/src/models/PseudoLSTM.h>
#include <auto_ml/src/models/TextClassifier.h>
#include <auto_ml/src/models/UniversalDeepTransformer.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::automl::python {

using models::ModelPipeline;
using models::QueryCandidateGenerator;
using models::TextClassifier;
using models::UniversalDeepTransformer;

void defineAutomlInModule(py::module_& module);

void createModelsSubmodule(py::module_& module);

void createUDTTypesSubmodule(py::module_& module);

void createUDTTemporalSubmodule(py::module_& module);

// Python wrappers for ModelPipline methods

deployment::UserInputMap createUserInputMap(const py::dict& parameters);

ModelPipeline createPipeline(const deployment::DeploymentConfigPtr& config,
                             const py::dict& parameters);

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters);

py::object predictTokensWrapper(ModelPipeline& model,
                                const std::vector<uint32_t>& tokens,
                                bool use_sparse_inference);

// UDT Factory
class UDTFactory {
 public:
  static QueryCandidateGenerator buildUDTGeneratorWrapper(
      py::object& obj, const std::string& source_column,
      const std::string& target_column, const std::string& dataset_size,
      char delimiter);

  static QueryCandidateGenerator buildUDTGeneratorWrapperTargetOnly(
      py::object& obj, const std::string& target_column,
      const std::string& dataset_size, char delimiter);

  static TextClassifier buildTextClassifier(py::object& obj,
                                            uint32_t input_vocab_size,
                                            uint32_t metadata_dim,
                                            uint32_t n_classes,
                                            const std::string& model_size);

  static UniversalDeepTransformer buildUDTClassifierWrapper(
      py::object& obj, data::ColumnDataTypes data_types,
      data::UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col, std::optional<uint32_t> n_target_classes,
      bool integer_target, std::string time_granularity, uint32_t lookahead,
      char delimiter, const std::optional<std::string>& model_config,
      const py::dict& options);

  static PseudoLSTM buildPseudoLSTM(
      py::object& obj, data::ColumnDataTypes data_types, std::string target_col,
      std::optional<uint32_t> n_target_classes, bool integer_target,
      uint32_t prediction_depth, char delimiter,
      const std::optional<std::string>& model_config, const py::dict& options

  );

  // These need to be here instead of inside UDTFactory because otherwise I was
  // getting weird linking errors
  static constexpr uint8_t UDT_GENERATOR_IDENTIFIER = 0;
  static constexpr uint8_t UDT_CLASSIFIER_IDENTIFIER = 1;
  static constexpr uint8_t UDT_TEXT_CLASSIFIER_IDENTIFIER = 2;
  static constexpr uint8_t UDT_PSEUDO_LSTM_IDENTIFIER = 3;

  static void save_classifier(const UniversalDeepTransformer& classifier,
                              const std::string& filename);

  static void save_generator(const QueryCandidateGenerator& generator,
                             const std::string& filename);

  static void saveTextClassifier(const TextClassifier& text_classifier,
                                 const std::string& filename);

  static py::object makeGeneratorInferenceTuple(
      std::vector<std::vector<std::string>> queries,
      std::vector<std::vector<float>> scores, bool return_scores) {
    if (return_scores) {
      return py::make_tuple(std::move(queries), std::move(scores));
    }
    return py::make_tuple(std::move(queries));
  }

  static py::object load(const std::string& filename);
};

}  // namespace thirdai::automl::python