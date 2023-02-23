#pragma once

#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/Generator.h>
#include <auto_ml/src/models/TextClassifier.h>
#include <auto_ml/src/udt/UDT.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::automl::python {

using models::QueryCandidateGenerator;
using models::TextClassifier;

void defineAutomlInModule(py::module_& module);

void createModelsSubmodule(py::module_& module);

void createUDTTypesSubmodule(py::module_& module);

void createUDTTemporalSubmodule(py::module_& module);

void createDeploymentSubmodule(py::module_& module);

// Python wrappers for ModelPipline methods

config::ArgumentMap createArgumentMap(const py::dict& input_args);

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

  static std::shared_ptr<udt::UDT> buildUDT(
      py::object& obj, data::ColumnDataTypes data_types,
      const data::UserProvidedTemporalRelationships&
          temporal_tracking_relationships,
      const std::string& target_col, std::optional<uint32_t> n_target_classes,
      bool integer_target, std::string time_granularity, uint32_t lookahead,
      char delimiter, const std::optional<std::string>& model_config,
      const py::dict& options);

  // These need to be here instead of inside UDTFactory because otherwise I was
  // getting weird linking errors
  static constexpr uint8_t UDT_GENERATOR_IDENTIFIER = 0;
  static constexpr uint8_t UDT_IDENTIFIER = 1;
  static constexpr uint8_t UDT_TEXT_CLASSIFIER_IDENTIFIER = 2;

  static void save_udt(const udt::UDT& classifier, const std::string& filename);

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