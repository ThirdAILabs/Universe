#include "DeploymentPython.h"
#include "DeploymentDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/python_bindings/UniversalDeepTransformerDocs.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/BlockConfig.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/ModelConfig.h>
#include <auto_ml/src/deployment_config/NodeConfig.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <auto_ml/src/deployment_config/dataset_configs/SingleBlockDatasetFactory.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/DataTypes.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/TemporalContext.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/UDTConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/UDTDatasetFactory.h>
#include <auto_ml/src/prebuilt_pipelines/UniversalDeepTransformer.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <search/src/Generator.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

namespace thirdai::automl::deployment::python {

void createDeploymentSubmodule(py::module_& thirdai_module) {
  py::module_ submodule = thirdai_module.def_submodule("deployment");

  py::class_<HyperParameter<uint32_t>, HyperParameterPtr<uint32_t>>(  // NOLINT
      submodule, "UintHyperParameter", docs::UINT_HYPERPARAMETER);

  py::class_<HyperParameter<float>, HyperParameterPtr<float>>(  // NOLINT
      submodule, "FloatHyperParameter", docs::FLOAT_HYPERPARAMETER);

  py::class_<HyperParameter<std::string>,  // NOLINT
             HyperParameterPtr<std::string>>(submodule, "StrHyperParameter",
                                             docs::STR_HYPERPARAMETER);

  py::class_<HyperParameter<bool>, HyperParameterPtr<bool>>(  // NOLINT
      submodule, "BoolHyperParameter", docs::BOOL_HYPERPARAMETER);

  py::class_<HyperParameter<bolt::SamplingConfigPtr>,  // NOLINT
             HyperParameterPtr<bolt::SamplingConfigPtr>>(
      submodule, "SamplingConfigHyperParameter", docs::STR_HYPERPARAMETER);

  py::class_<HyperParameter<UDTConfigPtr>,  // NOLINT
             HyperParameterPtr<UDTConfigPtr>>(submodule,
                                              "UDTConfigHyperParameter");

  /**
   * Do not change the order of these overloads. Because bool is a sublclass of
   * int in python, it must be declared first or calling this function with a
   * bool will result in the uint32_t function being called. Pybind guarentees
   * that overloads are tried in the order they were registered so this is safe
   * to do.
   */
  defConstantParameter<bool>(submodule, /* add_docs= */ true);
  defConstantParameter<uint32_t>(submodule, /* add_docs= */ false);
  defConstantParameter<float>(submodule, /* add_docs= */ false);
  defConstantParameter<std::string>(submodule, /* add_docs= */ false);
  defConstantParameter<bolt::SamplingConfigPtr>(submodule,
                                                /* add_docs= */ false);
  defConstantParameter<UDTConfigPtr>(submodule,
                                     /* add_docs= */ false);

  defOptionMappedParameter<bool>(submodule, /* add_docs= */ true);
  defOptionMappedParameter<uint32_t>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<float>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<std::string>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<bolt::SamplingConfigPtr>(submodule,
                                                    /* add_docs= */ false);
  defOptionMappedParameter<UDTConfigPtr>(submodule,
                                         /* add_docs= */ false);

  submodule.def("UserSpecifiedParameter", &makeUserSpecifiedParameter,
                py::arg("name"), py::arg("type"),
                docs::USER_SPECIFIED_PARAMETER);

  py::class_<AutotunedSparsityParameter, HyperParameter<float>,
             std::shared_ptr<AutotunedSparsityParameter>>(
      submodule, "AutotunedSparsityParameter")
      .def(py::init<std::string>(), py::arg("dimension_param_name"),
           docs::AUTOTUNED_SPARSITY_PARAMETER_INIT);

  py::class_<DatasetLabelDimensionParameter, HyperParameter<uint32_t>,
             std::shared_ptr<DatasetLabelDimensionParameter>>(
      submodule, "DatasetLabelDimensionParameter",
      docs::DATASET_LABEL_DIM_PARAM)
      .def(py::init<>())
      // This is why we pass in a py::object:
      // https://stackoverflow.com/questions/70504125/pybind11-pyclass-def-property-readonly-static-incompatible-function-arguments
      .def_property_readonly_static(
          "dimension_param_name", [](py::object& param) {
            (void)param;
            return DatasetLabelDimensionParameter::PARAM_NAME;
          });

  py::class_<NodeConfig, NodeConfigPtr>(submodule, "NodeConfig",  // NOLINT
                                        docs::NODE_CONFIG);

  py::class_<FullyConnectedNodeConfig, NodeConfig,
             std::shared_ptr<FullyConnectedNodeConfig>>(
      submodule, "FullyConnectedNodeConfig")
      .def(
          py::init<std::string, HyperParameterPtr<uint32_t>,
                   HyperParameterPtr<float>, HyperParameterPtr<std::string>,
                   std::string,
                   std::optional<HyperParameterPtr<bolt::SamplingConfigPtr>>>(),
          py::arg("name"), py::arg("dim"), py::arg("sparsity"),
          py::arg("activation"), py::arg("predecessor"),
          py::arg("sampling_config") = std::nullopt,
          docs::FULLY_CONNECTED_CONFIG_INIT_WITH_SPARSITY)
      .def(py::init<std::string, HyperParameterPtr<uint32_t>,
                    HyperParameterPtr<std::string>, std::string>(),
           py::arg("name"), py::arg("dim"), py::arg("activation"),
           py::arg("predecessor"), docs::FULLY_CONNECTED_CONFIG_INIT_DENSE);

  py::class_<ModelConfig, ModelConfigPtr>(submodule, "ModelConfig")
      .def(py::init<std::vector<std::string>, std::vector<NodeConfigPtr>,
                    std::shared_ptr<bolt::LossFunction>>(),
           py::arg("input_names"), py::arg("nodes"), py::arg("loss"),
           docs::MODEL_CONFIG_INIT);

  py::class_<BlockConfig, BlockConfigPtr>(submodule, "BlockConfig",  // NOLINT
                                          docs::BLOCK_CONFIG);

  py::class_<NumericalCategoricalBlockConfig, BlockConfig,
             std::shared_ptr<NumericalCategoricalBlockConfig>>(
      submodule, "NumericalCategoricalBlockConfig")
      .def(py::init<HyperParameterPtr<uint32_t>,
                    HyperParameterPtr<std::string>>(),
           py::arg("n_classes"), py::arg("delimiter"),
           docs::NUMERICAL_CATEGORICAL_BLOCK_CONFIG_INIT);

  py::class_<DenseArrayBlockConfig, BlockConfig,
             std::shared_ptr<DenseArrayBlockConfig>>(submodule,
                                                     "DenseArrayBlockConfig")
      .def(py::init<HyperParameterPtr<uint32_t>>(), py::arg("dim"),
           docs::DENSE_ARRAY_BLOCK_CONFIG_INIT);

  py::class_<TextBlockConfig, BlockConfig, std::shared_ptr<TextBlockConfig>>(
      submodule, "TextBlockConfig")
      .def(py::init<bool, HyperParameterPtr<uint32_t>>(),
           py::arg("use_pairgrams"), py::arg("range"),
           docs::TEXT_BLOCK_CONFIG_INIT_WITH_RANGE)
      .def(py::init<bool>(), py::arg("use_pairgrams"),
           docs::TEXT_BLOCK_CONFIG_INIT);

  py::class_<DatasetLoaderFactoryConfig,  // NOLINT
             DatasetLoaderFactoryConfigPtr>(
      submodule, "DatasetConfig", docs::DATASET_LOADER_FACTORY_CONFIG);

  py::class_<SingleBlockDatasetFactoryConfig, DatasetLoaderFactoryConfig,
             std::shared_ptr<SingleBlockDatasetFactoryConfig>>(
      submodule, "SingleBlockDatasetFactory")
      .def(py::init<BlockConfigPtr, BlockConfigPtr, HyperParameterPtr<bool>,
                    HyperParameterPtr<std::string>, bool>(),
           py::arg("data_block"), py::arg("label_block"), py::arg("shuffle"),
           py::arg("delimiter"), py::arg("has_header") = false,
           docs::SINGLE_BLOCK_DATASET_FACTORY_CONFIG_INIT);

  py::class_<UDTDatasetFactoryConfig, DatasetLoaderFactoryConfig,
             std::shared_ptr<UDTDatasetFactoryConfig>>(submodule,
                                                       "UDTDatasetFactory")
      .def(py::init<HyperParameterPtr<UDTConfigPtr>, HyperParameterPtr<bool>,
                    HyperParameterPtr<uint32_t>, HyperParameterPtr<bool>>(),
           py::arg("config"), py::arg("force_parallel"),
           py::arg("text_pairgram_word_limit"), py::arg("contextual_columns"));

  py::class_<TrainEvalParameters>(submodule, "TrainEvalParameters")
      .def(py::init<std::optional<uint32_t>, std::optional<uint32_t>, uint32_t,
                    bool, std::optional<float>>(),
           py::arg("rebuild_hash_tables_interval"),
           py::arg("reconstruct_hash_functions_interval"),
           py::arg("default_batch_size"), py::arg("freeze_hash_tables"),
           py::arg("prediction_threshold") = std::nullopt,
           docs::TRAIN_EVAL_PARAMETERS_CONFIG_INIT);

  py::class_<DeploymentConfig, DeploymentConfigPtr>(submodule,
                                                    "DeploymentConfig")
      .def(py::init<DatasetLoaderFactoryConfigPtr, ModelConfigPtr,
                    TrainEvalParameters>(),
           py::arg("dataset_config"), py::arg("model_config"),
           py::arg("train_eval_parameters"), docs::DEPLOYMENT_CONFIG_INIT)
      .def("save", &DeploymentConfig::save, py::arg("filename"),
           docs::DEPLOYMENT_CONFIG_SAVE)
      .def_static("load", &DeploymentConfig::load, py::arg("filename"),
                  docs::DEPLOYMENT_CONFIG_LOAD);

  py::class_<UDTDatasetFactory, UDTDatasetFactoryPtr>(submodule,
                                                      "TemporalContext")
      .def("reset", &UDTDatasetFactory::resetTemporalTrackers,
           docs::TEMPORAL_CONTEXT_RESET)
      .def("update_temporal_trackers",
           py::overload_cast<const LineInput&>(
               &UDTDatasetFactory::updateTemporalTrackers),
           py::arg("update"), docs::TEMPORAL_CONTEXT_UPDATE)
      .def("batch_update_temporal_trackers",
           py::overload_cast<const LineInputBatch&>(
               &UDTDatasetFactory::batchUpdateTemporalTrackers),
           py::arg("updates"), docs::TEMPORAL_CONTEXT_UPDATE_BATCH);
}

void createModelPipeline(py::module_& models_submodule) {
  py::class_<ModelPipeline>(models_submodule, "Pipeline")
      .def(py::init(&createPipeline), py::arg("deployment_config"),
           py::arg("parameters") = py::dict(),
           docs::MODEL_PIPELINE_INIT_FROM_CONFIG,
           bolt::python::OutputRedirect())
      .def(py::init(&createPipelineFromSavedConfig), py::arg("config_path"),
           py::arg("parameters") = py::dict(),
           docs::MODEL_PIPELINE_INIT_FROM_SAVED_CONFIG,
           bolt::python::OutputRedirect())
      .def("train", &trainOnFile<ModelPipeline>, py::arg("filename"),
           docs::MODEL_PIPELINE_TRAIN_FILE, bolt::python::OutputRedirect())
      .def("train", &trainOnDataLoader<ModelPipeline>, py::arg("data_source"),
           docs::MODEL_PIPELINE_TRAIN_DATA_LOADER,
           bolt::python::OutputRedirect())
      .def("evaluate", &evaluateOnFileWrapper<ModelPipeline>,
           py::arg("filename"), py::arg("eval_config") = std::nullopt,
           docs::MODEL_PIPELINE_EVALUATE_FILE, bolt::python::OutputRedirect())
      .def("evaluate", &evaluateOnDataLoaderWrapper, py::arg("data_source"),
           py::arg("eval_config") = std::nullopt,
           docs::MODEL_PIPELINE_EVALUATE_DATA_LOADER,
           bolt::python::OutputRedirect())
      .def("predict", &predictWrapper<ModelPipeline, LineInput>,
           py::arg("input_sample"), py::arg("use_sparse_inference") = false,
           docs::MODEL_PIPELINE_PREDICT)
      .def("explain", &ModelPipeline::explain<LineInput>,
           py::arg("input_sample"), py::arg("target_class") = std::nullopt,
           docs::MODEL_PIPELINE_EXPLAIN)
      .def("predict_tokens", &predictTokensWrapper, py::arg("tokens"),
           py::arg("use_sparse_inference") = false,
           docs::MODEL_PIPELINE_PREDICT_TOKENS)
      .def("predict_batch", &predictBatchWrapper<ModelPipeline, LineInputBatch>,
           py::arg("input_samples"), py::arg("use_sparse_inference") = false,
           docs::MODEL_PIPELINE_PREDICT_BATCH)
      .def("save", &ModelPipeline::save, py::arg("filename"),
           docs::MODEL_PIPELINE_SAVE)
      .def_static("load", &ModelPipeline::load, py::arg("filename"),
                  docs::MODEL_PIPELINE_LOAD)
      .def("get_data_processor", &ModelPipeline::getDataProcessor,
           docs::MODEL_PIPELINE_GET_DATA_PROCESSOR);
}

// These need to be here instead of inside UDTFactory because otherwise I was
// getting weird linking errors
static uint8_t const UDT_GENERATOR_IDENTIFIER = 0;
static uint8_t const UDT_CLASSIFIER_IDENTIFIER = 1;

class UDTFactory {
 public:
  static bolt::QueryCandidateGenerator buildUDTGeneratorWrapper(
      py::object& obj, const uint32_t& source_column_index,
      const uint32_t& target_column_index, const std::string& dataset_size) {
    (void)obj;
    return bolt::QueryCandidateGenerator::buildGeneratorFromDefaultConfig(
        /* source_column_index = */ source_column_index,
        /* target_column_index = */ target_column_index,
        /* dataset_size = */ dataset_size);
  }

  static UniversalDeepTransformer buildUDTClassifierWrapper(
      py::object& obj, ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col, uint32_t n_target_classes,
      bool integer_target = false, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::unordered_map<std::string, std::string>& options = {}) {
    (void)obj;
    return UniversalDeepTransformer::buildUDT(
        /* data_types = */ std::move(data_types),
        /* temporal_tracking_relationships = */
        std::move(temporal_tracking_relationships),
        /* target_col = */ std::move(target_col),
        /* n_target_classes = */ n_target_classes,
        /* integer_target = */ integer_target,
        /* time_granularity = */ std::move(time_granularity),
        /* lookahead = */ lookahead, /* delimiter = */ delimiter,
        /* options = */ options);
  }

  static void save_classifier(const UniversalDeepTransformer& classifier,
                              const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    filestream.write(reinterpret_cast<const char*>(&UDT_CLASSIFIER_IDENTIFIER),
                     1);
    classifier.save_stream(filestream);
  }

  static void save_generator(const bolt::QueryCandidateGenerator& generator,
                             const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    filestream.write(reinterpret_cast<const char*>(&UDT_GENERATOR_IDENTIFIER),
                     1);
    generator.save_stream(filestream);
  }

  static py::object load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    uint8_t first_byte;
    filestream.read(reinterpret_cast<char*>(&first_byte), 1);

    if (first_byte == UDT_GENERATOR_IDENTIFIER) {
      return py::cast(bolt::QueryCandidateGenerator::load_stream(filestream));
    }

    if (first_byte == UDT_CLASSIFIER_IDENTIFIER) {
      return py::cast(UniversalDeepTransformer::load_stream(filestream));
    }

    throw std::invalid_argument(
        "Found an invalid header byte in the saved file");
  }
};

void createUDTFactory(py::module_& bolt_submodule) {
  /**
   * This class definition overrides the __new__ method because we want to
   * modify class instantiation such that we can return objects of type
   * bolt.models.UDTGenerator or bolt.models.UDTClassifier instead of
   * an object of type bolt.UniversalDeepTransformer. Having this method
   * return a type other than that of the class on which it is being called
   * ensures that the __init__ method is never called.
   * https://stackoverflow.com/questions/26793600/decorate-call-with-staticmethod
   *
   */
  py::class_<UDTFactory>(bolt_submodule, "UniversalDeepTransformer",
                         docs::UDT_CLASS)
      .def("__new__", &UDTFactory::buildUDTClassifierWrapper,
           py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               UserProvidedTemporalRelationships(),
           py::arg("target"), py::arg("n_target_classes"),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("options") = OptionsMap(),
           docs::UDT_INIT, bolt::python::OutputRedirect())
      .def("__new__", &UDTFactory::buildUDTGeneratorWrapper,
           py::arg("source_column_index"), py::arg("target_column_index"),
           py::arg("dataset_size"), docs::UDT_GENERATOR_INIT)

      .def_static("load", &UDTFactory::load, py::arg("filename"),
                  docs::UDT_CLASSIFIER_AND_GENERATOR_LOAD);

  py::class_<ValidationOptions>(bolt_submodule, "Validation")
      .def(py::init<std::string, std::vector<std::string>,
                    std::optional<uint32_t>, bool>(),
           py::arg("filename"), py::arg("metrics"),
           py::arg("interval") = std::nullopt,
           py::arg("use_sparse_inference") = false);
}

void createUDTClassifierAndGenerator(py::module_& models_submodule) {
  py::class_<UDTConfig, UDTConfigPtr>(models_submodule, "UDTConfig")
      .def(py::init<ColumnDataTypes, UserProvidedTemporalRelationships,
                    std::string, uint32_t, bool, std::string, uint32_t, char>(),
           py::arg("data_types"), py::arg("temporal_tracking_relationships"),
           py::arg("target"), py::arg("n_target_classes"),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', docs::ORACLE_CONFIG_INIT,
           bolt::python::OutputRedirect());

  py::class_<UniversalDeepTransformer,
             std::shared_ptr<UniversalDeepTransformer>>(models_submodule,
                                                        "UDTClassifier")
      .def(py::init(&UniversalDeepTransformer::buildUDT), py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               UserProvidedTemporalRelationships(),
           py::arg("target"), py::arg("n_target_classes"),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("options") = OptionsMap(),
           docs::UDT_INIT, bolt::python::OutputRedirect())
      .def("train", &trainOnFile<UniversalDeepTransformer>, py::arg("filename"),
           docs::UDT_TRAIN, bolt::python::OutputRedirect())
      .def("class_name", &UniversalDeepTransformer::className,
           py::arg("neuron_id"), docs::UDT_CLASS_NAME)
      .def("evaluate", &evaluateOnFileWrapper<UniversalDeepTransformer>,
           py::arg("filename"), py::arg("eval_config") = std::nullopt,
           docs::UDT_EVALUATE, bolt::python::OutputRedirect())
      .def("predict", &predictWrapper<UniversalDeepTransformer, MapInput>,
           py::arg("input_sample"), py::arg("use_sparse_inference") = false,
           docs::UDT_PREDICT)
      .def("predict_batch",
           &predictBatchWrapper<UniversalDeepTransformer, MapInputBatch>,
           py::arg("input_samples"), py::arg("use_sparse_inference") = false,
           docs::UDT_PREDICT_BATCH)
      .def(
          "embedding_representation",
          [](UniversalDeepTransformer& model, const MapInput& input) {
            return convertBoltVectorToNumpy(
                model.embeddingRepresentation(input));
          },
          py::arg("input_sample"), docs::UDT_EMBEDDING_REPRESENTATION)
      .def("index", &UniversalDeepTransformer::updateTemporalTrackers,
           py::arg("input_sample"), docs::UDT_INDEX,
           bolt::python::OutputRedirect())
      .def("index_batch",
           &UniversalDeepTransformer::batchUpdateTemporalTrackers,
           py::arg("input_samples"), docs::UDT_INDEX_BATCH)
      .def("reset_temporal_trackers",
           &UniversalDeepTransformer::resetTemporalTrackers,
           docs::UDT_RESET_TEMPORAL_TRACKERS)
      .def("explain", &UniversalDeepTransformer::explain<MapInput>,
           py::arg("input_sample"), py::arg("target_class") = std::nullopt,
           docs::UDT_EXPLAIN)
      .def("save", &UDTFactory::save_classifier, py::arg("filename"),
           docs::UDT_SAVE);

  py::class_<bolt::QueryCandidateGenerator,
             std::shared_ptr<bolt::QueryCandidateGenerator>>(models_submodule,
                                                             "UDTGenerator")
      .def(py::init(
               &bolt::QueryCandidateGenerator::buildGeneratorFromDefaultConfig),
           py::arg("source_column_index"), py::arg("target_column_index"),
           py::arg("dataset_size"), docs::UDT_GENERATOR_INIT)
      .def("train", &bolt::QueryCandidateGenerator::buildFlashIndex,
           py::arg("filename"), docs::UDT_GENERATOR_TRAIN)
      .def("evaluate", &bolt::QueryCandidateGenerator::evaluateOnFile,
           py::arg("filename"), py::arg("top_k"), docs::UDT_GENERATOR_EVALUATE)
      .def(
          "predict",
          [](bolt::QueryCandidateGenerator& udt_generator_model,
             const std::string& sample, uint32_t top_k) {
            return udt_generator_model.queryFromList({sample}, top_k);
          },
          py::arg("query"), py::arg("top_k"), docs::UDT_GENERATOR_PREDICT)
      .def("predict_batch", &bolt::QueryCandidateGenerator::queryFromList,
           py::arg("queries"), py::arg("top_k"),
           docs::UDT_GENERATOR_PREDICT_BATCH)
      .def("save", &UDTFactory::save_generator, py::arg("filename"),
           docs::UDT_GENERATOR_SAVE);
}

template <typename T>
void defConstantParameter(py::module_& submodule, bool add_docs) {
  // Because this is an overloaded function, the docsstring will be rendered
  // for each overload. This option is to ensure that it can only be rendered
  // for the first one.
  const char* const docstring =
      add_docs ? docs::CONSTANT_PARAMETER : "See docs above.";

  submodule.def("ConstantParameter", &ConstantParameter<T>::make,
                py::arg("value").noconvert(), docstring);
}

template <typename T>
void defOptionMappedParameter(py::module_& submodule, bool add_docs) {
  // Because this is an overloaded function, the docsstring will be rendered
  // for each overload. This option is to ensure that it can only be rendered
  // for the first one.
  const char* const docstring =
      add_docs ? docs::OPTION_MAPPED_PARAMETER : "See docs above.";

  submodule.def("OptionMappedParameter", &OptionMappedParameter<T>::make,
                py::arg("option_name"), py::arg("values").noconvert(),
                docstring);
}

py::object makeUserSpecifiedParameter(const std::string& name,
                                      const py::object& type) {
  if (py::str(type).cast<std::string>() == "<class 'bool'>") {
    return py::cast(UserSpecifiedParameter<bool>::make(name));
  }

  if (py::str(type).cast<std::string>() == "<class 'int'>") {
    return py::cast(UserSpecifiedParameter<uint32_t>::make(name));
  }

  if (py::str(type).cast<std::string>() == "<class 'float'>") {
    return py::cast(UserSpecifiedParameter<float>::make(name));
  }

  if (py::str(type).cast<std::string>() == "<class 'str'>") {
    return py::cast(UserSpecifiedParameter<std::string>::make(name));
  }

  if (py::str(type).cast<std::string>() ==
      "<class 'thirdai._thirdai.bolt.models.UDTConfig'>") {
    return py::cast(UserSpecifiedParameter<UDTConfigPtr>::make(name));
  }

  throw std::invalid_argument("Invalid type '" +
                              py::str(type).cast<std::string>() +
                              "' passed to UserSpecifiedParameter. Must be one "
                              "of bool, int, float, str, or UDTConfig.");
}

ModelPipeline createPipeline(const DeploymentConfigPtr& config,
                             const py::dict& parameters) {
  UserInputMap cpp_parameters;
  for (const auto& [k, v] : parameters) {
    if (!py::isinstance<py::str>(k)) {
      throw std::invalid_argument("Keys of parameters map must be strings.");
    }
    std::string name = k.cast<std::string>();

    if (py::isinstance<py::bool_>(v)) {
      bool value = v.cast<bool>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else if (py::isinstance<py::int_>(v)) {
      uint32_t value = v.cast<uint32_t>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else if (py::isinstance<py::float_>(v)) {
      float value = v.cast<float>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else if (py::isinstance<py::str>(v)) {
      std::string value = v.cast<std::string>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else if (py::isinstance<UDTConfig>(v)) {
      UDTConfigPtr value = v.cast<UDTConfigPtr>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else {
      throw std::invalid_argument("Invalid type '" +
                                  py::str(v.get_type()).cast<std::string>() +
                                  "'. Values of parameters dictionary must be "
                                  "bool, int, float, str, or UDTConfig.");
    }
  }

  return ModelPipeline::make(config, cpp_parameters);
}

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters) {
  auto config = DeploymentConfig::load(config_path);

  return createPipeline(config, parameters);
}

template <typename Model>
void trainOnFile(Model& model, const std::string& filename,
                 const py::kwargs& kwargs) {
  uint32_t batch_size;
  if (kwargs.contains("batch_size")) {
    batch_size = kwargs["batch_size"].cast<uint32_t>();
  } else {
    batch_size = model.defaultBatchSize();
  }
  auto data_source = dataset::SimpleFileDataLoader::make(
      filename, /* target_batch_size= */ batch_size);

  trainOnDataLoader(model, data_source, kwargs);
}

template <typename Model>
void trainOnDataLoader(Model& model, const dataset::DataLoaderPtr& data_source,
                       const py::kwargs& kwargs) {
  TrainOptions train_options(data_source);

  // Parse kwargs

  for (const auto& [key, value] : kwargs) {
    std::string name = key.cast<std::string>();
    if (name == "learning_rate") {
      train_options.setLearningRate(value.cast<float>());
    } else if (name == "epochs") {
      train_options.setEpochs(kwargs["epochs"].cast<uint32_t>());
    } else if (name == "validation") {
      train_options.setValidation(
          kwargs["validation"].cast<ValidationOptions>());
    } else if (name == "callbacks") {
      train_options.setCallbacks(
          kwargs["callbacks"].cast<std::vector<bolt::CallbackPtr>>());
    } else if (name == "max_in_memory_batches") {
      train_options.setMaxInMemoryBatches(
          kwargs["max_in_memory_batches"].cast<uint32_t>());
    } else if (name != "batch_size") {
      // batch_size is a valid arg just not used here so we don't throw if it's
      // present.
      throw py::value_error("Unexpected argument '" + name +
                            "' passed to train.");
    }
  }

  model.train(train_options);
}

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::EvalConfig>& eval_config) {
  auto output = model.evaluate(data_source, eval_config);

  return convertInferenceTrackerToNumpy(output);
}

template <typename Model>
py::object evaluateOnFileWrapper(Model& model, const std::string& filename,
                                 std::optional<bolt::EvalConfig>& eval_config) {
  return evaluateOnDataLoaderWrapper(model,
                                     dataset::SimpleFileDataLoader::make(
                                         filename, DEFAULT_EVALUATE_BATCH_SIZE),
                                     eval_config);
}

template <typename Model, typename InputType>
py::object predictWrapper(Model& model, const InputType& sample,
                          bool use_sparse_inference) {
  BoltVector output =
      model.template predict<InputType>(sample, use_sparse_inference);
  return convertBoltVectorToNumpy(output);
}

py::object predictTokensWrapper(ModelPipeline& model,
                                const std::vector<uint32_t>& tokens,
                                bool use_sparse_inference) {
  std::stringstream sentence;
  for (uint32_t i = 0; i < tokens.size(); i++) {
    if (i > 0) {
      sentence << ' ';
    }
    sentence << tokens[i];
  }
  return predictWrapper(model, sentence.str(), use_sparse_inference);
}

template <typename Model, typename InputBatchType>
py::object predictBatchWrapper(Model& model, const InputBatchType& samples,
                               bool use_sparse_inference) {
  BoltBatch outputs = model.template predictBatch<InputBatchType>(
      samples, use_sparse_inference);

  return convertBoltBatchToNumpy(outputs);
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object convertInferenceTrackerToNumpy(
    bolt::InferenceOutputTracker& output) {
  uint32_t num_samples = output.numSamples();
  uint32_t inference_dim = output.numNonzerosInOutput();

  const uint32_t* active_neurons_ptr = output.getNonowningActiveNeuronPointer();
  const float* activations_ptr = output.getNonowningActivationPointer();

  py::object output_handle = py::cast(std::move(output));

  NumpyArray<float> activations_array(
      /* shape= */ {num_samples, inference_dim},
      /* strides= */ {inference_dim * sizeof(float), sizeof(float)},
      /* ptr= */ activations_ptr, /* base= */ output_handle);

  if (!active_neurons_ptr) {
    return py::object(std::move(activations_array));
  }

  // See comment above activations_array for the python memory reasons behind
  // passing in active_neuron_handle
  NumpyArray<uint32_t> active_neurons_array(
      /* shape= */ {num_samples, inference_dim},
      /* strides= */ {inference_dim * sizeof(uint32_t), sizeof(uint32_t)},
      /* ptr= */ active_neurons_ptr, /* base= */ output_handle);

  return py::make_tuple(std::move(activations_array),
                        std::move(active_neurons_array));
}

py::object convertBoltVectorToNumpy(const BoltVector& vector) {
  NumpyArray<float> activations_array(vector.len);
  std::copy(vector.activations, vector.activations + vector.len,
            activations_array.mutable_data());

  if (vector.isDense()) {
    return py::object(std::move(activations_array));
  }

  NumpyArray<uint32_t> active_neurons_array(vector.len);
  std::copy(vector.active_neurons, vector.active_neurons + vector.len,
            active_neurons_array.mutable_data());

  return py::make_tuple(active_neurons_array, activations_array);
}

py::object convertBoltBatchToNumpy(const BoltBatch& batch) {
  uint32_t length = batch[0].len;

  NumpyArray<float> activations_array(
      /* shape= */ {batch.getBatchSize(), length});

  std::optional<NumpyArray<uint32_t>> active_neurons_array = std::nullopt;
  if (!batch[0].isDense()) {
    active_neurons_array =
        NumpyArray<uint32_t>(/* shape= */ {batch.getBatchSize(), length});
  }

  for (uint32_t i = 0; i < batch.getBatchSize(); i++) {
    if (batch[i].len != length) {
      throw std::invalid_argument(
          "Cannot convert BoltBatch without constant lengths to a numpy "
          "array.");
    }
    if (batch[i].isDense() != !active_neurons_array.has_value()) {
      throw std::invalid_argument(
          "Cannot convert BoltBatch without constant sparsity to a numpy "
          "array.");
    }

    std::copy(batch[i].activations, batch[i].activations + length,
              activations_array.mutable_data() + i * length);
    if (active_neurons_array) {
      std::copy(batch[i].active_neurons, batch[i].active_neurons + length,
                active_neurons_array->mutable_data() + i * length);
    }
  }

  if (active_neurons_array) {
    return py::make_tuple(std::move(active_neurons_array.value()),
                          std::move(activations_array));
  }
  return py::object(std::move(activations_array));
}

}  // namespace thirdai::automl::deployment::python