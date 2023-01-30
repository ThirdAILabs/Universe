#include "AutomlPython.h"
#include "AutomlDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/dataset_factories/DatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <auto_ml/src/models/RNN.h>
#include <auto_ml/src/models/UniversalDeepTransformer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/detail/common.h>
#include <pybind11/pytypes.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::python {

void defineAutomlInModule(py::module_& module) {
  py::class_<models::ValidationOptions>(module, "Validation")
      .def(py::init<std::string, std::vector<std::string>,
                    std::optional<uint32_t>, bool>(),
           py::arg("filename"), py::arg("metrics"),
           py::arg("interval") = std::nullopt,
           py::arg("use_sparse_inference") = false, docs::VALIDATION);

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
  py::class_<UDTFactory>(module, "UniversalDeepTransformer", docs::UDT_CLASS)
      .def("__new__", &UDTFactory::buildUDTClassifierWrapper,
           py::arg("data_types"), py::arg("target"),
           py::arg("n_target_classes") = std::nullopt,
           py::arg("temporal_tracking_relationships") =
               data::UserProvidedTemporalRelationships(),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("model_config") = std::nullopt,
           py::arg("options") = py::dict(), docs::UDT_INIT,
           bolt::python::OutputRedirect())
      .def("__new__", &UDTFactory::buildUDTGeneratorWrapper,
           py::arg("source_column"), py::arg("target_column"),
           py::arg("dataset_size"), py::arg("delimiter") = ',',
           docs::UDT_GENERATOR_INIT)
      .def("__new__", &UDTFactory::buildUDTGeneratorWrapperTargetOnly,
           py::arg("target_column"), py::arg("dataset_size"),
           py::arg("delimiter") = ',', docs::UDT_GENERATOR_INIT)
      .def("__new__", &UDTFactory::buildTextClassifier,
           py::arg("input_vocab_size"), py::arg("metadata_dim"),
           py::arg("n_classes"), py::arg("model_size"),
           docs::TEXT_CLASSIFIER_INIT)
      .def_static("load", &UDTFactory::load, py::arg("filename"),
                  docs::UDT_CLASSIFIER_AND_GENERATOR_LOAD);
}

void createModelsSubmodule(py::module_& module) {
  auto models_submodule = module.def_submodule("models");

  py::class_<ModelPipeline, std::shared_ptr<ModelPipeline>>(models_submodule,
                                                            "Pipeline")
      .def("train_with_source", &ModelPipeline::train, py::arg("data_source"),
           py::arg("train_config"), py::arg("validation") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt,
           py::arg("batch_size") = std::nullopt,
           docs::MODEL_PIPELINE_TRAIN_DATA_SOURCE,
           bolt::python::OutputRedirect())
      .def("evaluate_with_source", &ModelPipeline::evaluate,
           py::arg("data_source"), py::arg("eval_config") = std::nullopt,
           py::arg("return_predicted_class") = false,
           py::arg("return_metrics") = false,
           docs::MODEL_PIPELINE_EVALUATE_DATA_SOURCE,
           bolt::python::OutputRedirect())
      .def("save", &ModelPipeline::save, py::arg("filename"),
           docs::MODEL_PIPELINE_SAVE)
      .def_static("load", &ModelPipeline::load, py::arg("filename"),
                  docs::MODEL_PIPELINE_LOAD)
      .def("get_data_processor", &ModelPipeline::getDataProcessor,
           docs::MODEL_PIPELINE_GET_DATA_PROCESSOR)
      .def("_get_model", &ModelPipeline::getModel)
      .def("_set_model", &ModelPipeline::setModel, py::arg("trained_model"))
      .def_property_readonly("default_train_batch_size",
                             &ModelPipeline::defaultBatchSize)
      .def_property_readonly_static(
          "default_evaluate_batch_size", [](const py::object& /* self */) {
            return models::DEFAULT_EVALUATE_BATCH_SIZE;
          });

  py::class_<data::UDTDatasetFactory, data::UDTDatasetFactoryPtr>(
      models_submodule, "TemporalContext")
      .def("get_dataset_loader",
           &data::UDTDatasetFactory::getLabeledDatasetLoader,
           py::arg("data_source"), py::arg("training"))
      .def("reset", &data::UDTDatasetFactory::resetTemporalTrackers,
           docs::TEMPORAL_CONTEXT_RESET)
      .def("update_temporal_trackers",
           py::overload_cast<const MapInput&>(
               &data::UDTDatasetFactory::updateTemporalTrackers),
           py::arg("update"), docs::TEMPORAL_CONTEXT_UPDATE)
      .def("batch_update_temporal_trackers",
           py::overload_cast<const MapInputBatch&>(
               &data::UDTDatasetFactory::batchUpdateTemporalTrackers),
           py::arg("updates"), docs::TEMPORAL_CONTEXT_UPDATE_BATCH)
      .def("verify_can_distribute",
           &data::UDTDatasetFactory::verifyCanDistribute)
      .def(bolt::python::getPickleFunction<data::UDTDatasetFactory>());

  py::class_<data::UDTConfig, data::UDTConfigPtr>(models_submodule, "UDTConfig")
      .def(py::init<data::ColumnDataTypes,
                    data::UserProvidedTemporalRelationships, std::string,
                    uint32_t, bool, std::string, uint32_t, char>(),
           py::arg("data_types"), py::arg("temporal_tracking_relationships"),
           py::arg("target"), py::arg("n_target_classes"),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', docs::UDT_CONFIG_INIT,
           bolt::python::OutputRedirect());

  py::class_<UniversalDeepTransformer, ModelPipeline,
             std::shared_ptr<UniversalDeepTransformer>>(models_submodule,
                                                        "UDTClassifier")
      .def("class_name", &UniversalDeepTransformer::className,
           py::arg("neuron_id"), docs::UDT_CLASS_NAME)
      .def("predict",
           py::overload_cast<const MapInput&, bool, bool>(
               &UniversalDeepTransformer::predict),
           py::arg("input_sample"), py::arg("use_sparse_inference") = false,
           py::arg("return_predicted_class") = false, docs::UDT_PREDICT)
      .def("predict_batch",
           py::overload_cast<const MapInputBatch&, bool, bool>(
               &UniversalDeepTransformer::predictBatch),
           py::arg("input_samples"), py::arg("use_sparse_inference") = false,
           py::arg("return_predicted_class") = false, docs::UDT_PREDICT_BATCH)
      .def("cold_start", &UniversalDeepTransformer::coldStartPretraining,
           py::arg("dataset"), py::arg("strong_column_names"),
           py::arg("weak_column_names"), py::arg("learning_rate"))
      .def(
          "embedding_representation",
          [](UniversalDeepTransformer& model, const MapInput& input) {
            return models::convertBoltVectorToNumpy(
                model.embeddingRepresentation(input));
          },
          py::arg("input_sample"), docs::UDT_EMBEDDING_REPRESENTATION)
      .def("get_prediction_threshold",
           &UniversalDeepTransformer::getPredictionThreshold)
      .def("set_prediction_threshold",
           &UniversalDeepTransformer::setPredictionThreshold,
           py::arg("threshold"))
      .def("index", &UniversalDeepTransformer::updateTemporalTrackers,
           py::arg("input_sample"), docs::UDT_INDEX,
           bolt::python::OutputRedirect())
      .def("index_batch",
           &UniversalDeepTransformer::batchUpdateTemporalTrackers,
           py::arg("input_samples"), docs::UDT_INDEX_BATCH)
      .def("index_metadata", &UniversalDeepTransformer::updateMetadata,
           py::arg("column_name"), py::arg("update"), docs::UDT_INDEX_METADATA,
           bolt::python::OutputRedirect())
      .def("index_metadata_batch",
           &UniversalDeepTransformer::updateMetadataBatch,
           py::arg("column_name"), py::arg("updates"),
           docs::UDT_INDEX_METADATA_BATCH, bolt::python::OutputRedirect())
      .def("reset_temporal_trackers",
           &UniversalDeepTransformer::resetTemporalTrackers,
           docs::UDT_RESET_TEMPORAL_TRACKERS)
      .def("explain", &UniversalDeepTransformer::explain,
           py::arg("input_sample"), py::arg("target_class") = std::nullopt,
           docs::UDT_EXPLAIN)
      .def("save", &UDTFactory::save_classifier, py::arg("filename"),
           docs::UDT_SAVE);

  py::class_<QueryCandidateGenerator, std::shared_ptr<QueryCandidateGenerator>>(
      models_submodule, "UDTGenerator")
      .def(py::init(&QueryCandidateGenerator::buildGeneratorFromDefaultConfig),
           py::arg("source_column"), py::arg("target_column"),
           py::arg("dataset_size"), py::arg("delimiter") = ',',
           docs::UDT_GENERATOR_INIT)
      .def("train", &QueryCandidateGenerator::train, py::arg("filename"),
           py::arg("use_supervised") = true, docs::UDT_GENERATOR_TRAIN)
      .def(
          "evaluate",
          [](QueryCandidateGenerator& udt_generator_model,
             const std::string& filename, uint32_t top_k, bool return_scores) {
            auto [reformulated_queries, scores] =
                udt_generator_model.evaluateOnFile(filename, top_k);
            return UDTFactory::makeGeneratorInferenceTuple(
                reformulated_queries, scores, return_scores);
          },
          py::arg("filename"), py::arg("top_k"),
          py::arg("return_scores") = false, docs::UDT_GENERATOR_EVALUATE)
      .def(
          "predict",
          [](QueryCandidateGenerator& udt_generator_model,
             const std::string& sample, uint32_t top_k, bool return_scores) {
            auto [reformulated_queries, scores] =
                udt_generator_model.queryFromList({sample}, top_k);
            return UDTFactory::makeGeneratorInferenceTuple(
                reformulated_queries, scores, return_scores);
          },
          py::arg("query"), py::arg("top_k"), py::arg("return_scores") = false,
          docs::UDT_GENERATOR_PREDICT)
      .def(
          "predict_batch",
          [](QueryCandidateGenerator& udt_generator_model,
             const std::vector<std::string>& queries, uint32_t top_k,
             bool return_scores) {
            auto [reformulated_queries, scores] =
                udt_generator_model.queryFromList(queries, top_k);
            return UDTFactory::makeGeneratorInferenceTuple(
                reformulated_queries, scores, return_scores);
          },
          py::arg("queries"), py::arg("top_k"),
          py::arg("return_scores") = false, docs::UDT_GENERATOR_PREDICT_BATCH)
      .def("save", &UDTFactory::save_generator, py::arg("filename"),
           docs::UDT_GENERATOR_SAVE);

  py::class_<TextClassifier, std::shared_ptr<TextClassifier>>(
      models_submodule, "UDTTextClassifier")
      .def("train", &TextClassifier::trainOnBatch, py::arg("data"),
           py::arg("labels"), py::arg("learning_rate"),
           docs::TEXT_CLASSIFIER_TRAIN)
      .def("validate", &TextClassifier::validateOnBatch, py::arg("data"),
           py::arg("labels"), docs::TEXT_CLASSIFIER_VALIDATE)
      .def("predict", &TextClassifier::predict, py::arg("data"),
           docs::TEXT_CLASSIFIER_PREDICT)
      .def("save", &UDTFactory::saveTextClassifier, py::arg("filename"),
           docs::TEXT_CLASSIFIER_SAVE);

  py::class_<RNN, ModelPipeline, std::shared_ptr<RNN>>(models_submodule,
                                                       "UDTRNN")
      .def("predict", &RNN::predict, py::arg("input_sample"),
           py::arg("use_sparse_inference") = false,
           py::arg("return_predicted_class") = true, docs::UDT_PREDICT)
      .def("predict_batch", &RNN::predictBatch, py::arg("input_samples"),
           py::arg("use_sparse_inference") = false,
           py::arg("return_predicted_class") = true, docs::UDT_PREDICT_BATCH)
      .def("index_metadata", &RNN::updateMetadata, py::arg("column_name"),
           py::arg("update"), docs::UDT_INDEX_METADATA,
           bolt::python::OutputRedirect())
      .def("index_metadata_batch", &RNN::updateMetadataBatch,
           py::arg("column_name"), py::arg("updates"),
           docs::UDT_INDEX_METADATA_BATCH, bolt::python::OutputRedirect())
      .def("explain", &RNN::explain, py::arg("input_sample"),
           py::arg("target_class") = std::nullopt, docs::UDT_EXPLAIN)
      .def("save", &UDTFactory::saveRNN, py::arg("filename"), docs::UDT_SAVE);
}

void createUDTTypesSubmodule(py::module_& module) {
  auto udt_types_submodule = module.def_submodule("types");

  py::class_<automl::data::DataType,
             automl::data::DataTypePtr>(  // NOLINT
      udt_types_submodule, "ColumnType", "Base class for bolt types.")
      .def("__str__", &automl::data::DataType::toString)
      .def("__repr__", &automl::data::DataType::toString);

  py::class_<automl::data::CategoricalMetadataConfig,
             automl::data::CategoricalMetadataConfigPtr>(udt_types_submodule,
                                                         "metadata")
      .def(py::init<std::string, std::string, automl::data::ColumnDataTypes,
                    char>(),
           py::arg("filename"), py::arg("key_column_name"),
           py::arg("data_types"), py::arg("delimiter") = ',',
           docs::UDT_CATEGORICAL_METADATA_CONFIG);

  py::class_<automl::data::CategoricalDataType, automl::data::DataType,
             automl::data::CategoricalDataTypePtr>(udt_types_submodule,
                                                   "categorical")
      .def(py::init<std::optional<char>,
                    automl::data::CategoricalMetadataConfigPtr>(),
           py::arg("delimiter") = std::nullopt, py::arg("metadata") = nullptr,
           docs::UDT_CATEGORICAL_TEMPORAL);

  py::class_<automl::data::NumericalDataType, automl::data::DataType,
             automl::data::NumericalDataTypePtr>(udt_types_submodule,
                                                 "numerical")
      .def(py::init<std::pair<double, double>, std::string>(), py::arg("range"),
           py::arg("granularity") = "m", docs::UDT_NUMERICAL_TYPE);

  py::class_<automl::data::TextDataType, automl::data::DataType,
             automl::data::TextDataTypePtr>(udt_types_submodule, "text")
      .def(py::init<std::optional<double>, bool>(),
           py::arg("average_n_words") = std::nullopt,
           py::arg("use_attention") = false, docs::UDT_TEXT_TYPE);

  py::class_<automl::data::DateDataType, automl::data::DataType,
             automl::data::DateDataTypePtr>(udt_types_submodule, "date")
      .def(py::init<>(), docs::UDT_DATE_TYPE);

  py::class_<automl::data::SequenceDataType, automl::data::DataType,
             automl::data::SequenceDataTypePtr>(udt_types_submodule, "sequence")
      .def(py::init<char, std::optional<uint32_t>>(),
           py::arg("delimiter") = ' ', py::arg("max_length") = std::nullopt,
           docs::UDT_SEQUENCE_TYPE);
}

void createUDTTemporalSubmodule(py::module_& module) {
  auto udt_temporal_submodule = module.def_submodule("temporal");

  py::class_<automl::data::TemporalConfig>(  // NOLINT
      udt_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  udt_temporal_submodule.def(
      "categorical", automl::data::TemporalConfig::categorical,
      py::arg("column_name"), py::arg("track_last_n"),
      py::arg("column_known_during_inference") = false,
      py::arg("use_metadata") = false, docs::UDT_CATEGORICAL_TEMPORAL);

  udt_temporal_submodule.def("numerical",
                             automl::data::TemporalConfig::numerical,
                             py::arg("column_name"), py::arg("history_length"),
                             py::arg("column_known_during_inference") = false,
                             docs::UDT_NUMERICAL_TEMPORAL);
}

void createDeploymentSubmodule(py::module_& module) {
#if THIRDAI_EXPOSE_ALL

  auto deployment = module.def_submodule("deployment");

  deployment.def("load_config", &config::loadConfig, py::arg("filename"));

  deployment.def("dump_config", &config::dumpConfig, py::arg("config"),
                 py::arg("filename"));

  deployment.def(
      "load_model_from_config",
      [](const std::string& config_file, const py::dict& parameters,
         const std::vector<uint32_t>& input_dims) {
        auto json_config = json::parse(config::loadConfig(config_file));
        auto user_input = createArgumentMap(parameters);

        return config::buildModel(json_config, user_input, input_dims);
      },
      py::arg("config_file"), py::arg("parameters"), py::arg("input_dims"));

#else
  (void)module;

#endif
}

config::ArgumentMap createArgumentMap(const py::dict& input_args) {
  config::ArgumentMap args;
  for (const auto& [k, v] : input_args) {
    if (!py::isinstance<py::str>(k)) {
      throw std::invalid_argument("Keys of parameters map must be strings.");
    }
    std::string name = k.cast<std::string>();

    if (py::isinstance<py::bool_>(v)) {
      bool value = v.cast<bool>();
      args.insert(name, value);
    } else if (py::isinstance<py::int_>(v)) {
      uint32_t value = v.cast<uint32_t>();
      args.insert(name, value);
    } else if (py::isinstance<py::float_>(v)) {
      float value = v.cast<float>();
      args.insert(name, value);
    } else if (py::isinstance<py::str>(v)) {
      std::string value = v.cast<std::string>();
      args.insert(name, value);
    } else {
      throw std::invalid_argument("Invalid type '" +
                                  py::str(v.get_type()).cast<std::string>() +
                                  "'. Values of parameters dictionary must be "
                                  "bool, int, float, str, or UDTConfig.");
    }
  }

  return args;
}

// UDT Factory Methods

QueryCandidateGenerator UDTFactory::buildUDTGeneratorWrapper(
    py::object& obj, const std::string& source_column,
    const std::string& target_column, const std::string& dataset_size,
    char delimiter) {
  (void)obj;
  return QueryCandidateGenerator::buildGeneratorFromDefaultConfig(
      /* source_column_name = */ source_column,
      /* target_column_name = */ target_column,
      /* dataset_size = */ dataset_size,
      /* delimiter = */ delimiter);
}

QueryCandidateGenerator UDTFactory::buildUDTGeneratorWrapperTargetOnly(
    py::object& obj, const std::string& target_column,
    const std::string& dataset_size, char delimiter) {
  (void)obj;
  return QueryCandidateGenerator::buildGeneratorFromDefaultConfig(
      /* source_column_name = */ target_column,
      /* target_column_name = */ target_column,
      /* dataset_size = */ dataset_size, /* delimiter = */ delimiter);
}

TextClassifier UDTFactory::buildTextClassifier(py::object& obj,
                                               uint32_t input_vocab_size,
                                               uint32_t metadata_dim,
                                               uint32_t n_classes,
                                               const std::string& model_size) {
  (void)obj;
  return TextClassifier(input_vocab_size, metadata_dim, n_classes, model_size);
}

py::object UDTFactory::buildUDTClassifierWrapper(
    py::object& obj, data::ColumnDataTypes data_types, std::string target_col,
    std::optional<uint32_t> n_target_classes,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const py::dict& options) {
  (void)obj;
  if (!data_types.count(target_col)) {
    throw std::invalid_argument("data_types must contain target_col.");
  }
  if (data::asSequence(data_types[target_col])) {
    if (!n_target_classes) {
      throw std::invalid_argument(
          "Must provide n_target_classes when target is a sequence.");
    }
    return py::cast(RNN::buildRNN(std::move(data_types), std::move(target_col),
                                  *n_target_classes));
  }
  return py::cast(UniversalDeepTransformer::buildUDT(
      /* data_types = */ std::move(data_types),
      /* temporal_tracking_relationships = */
      std::move(temporal_tracking_relationships),
      /* target_col = */ std::move(target_col),
      /* n_target_classes = */ n_target_classes,
      /* integer_target = */ integer_target,
      /* time_granularity = */ std::move(time_granularity),
      /* lookahead = */ lookahead, /* delimiter = */ delimiter,
      /* model_config= */ model_config,
      /* options = */ createArgumentMap(options)));
}

void UDTFactory::save_classifier(const UniversalDeepTransformer& classifier,
                                 const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(reinterpret_cast<const char*>(&UDT_CLASSIFIER_IDENTIFIER),
                   1);
  classifier.save_stream(filestream);
}

void UDTFactory::save_generator(const QueryCandidateGenerator& generator,
                                const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(reinterpret_cast<const char*>(&UDT_GENERATOR_IDENTIFIER), 1);
  generator.save_stream(filestream);
}

void UDTFactory::saveTextClassifier(const TextClassifier& text_classifier,
                                    const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(
      reinterpret_cast<const char*>(&UDT_TEXT_CLASSIFIER_IDENTIFIER), 1);
  text_classifier.save_stream(filestream);
}

void UDTFactory::saveRNN(const RNN& rnn, const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(reinterpret_cast<const char*>(&UDT_RNN_IDENTIFIER), 1);
  rnn.save_stream(filestream);
}

py::object UDTFactory::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  uint8_t first_byte;
  filestream.read(reinterpret_cast<char*>(&first_byte), 1);

  if (first_byte == UDT_GENERATOR_IDENTIFIER) {
    return py::cast(QueryCandidateGenerator::load_stream(filestream));
  }

  if (first_byte == UDT_CLASSIFIER_IDENTIFIER) {
    return py::cast(UniversalDeepTransformer::load_stream(filestream));
  }

  if (first_byte == UDT_TEXT_CLASSIFIER_IDENTIFIER) {
    return py::cast(TextClassifier::load_stream(filestream));
  }

  if (first_byte == UDT_RNN_IDENTIFIER) {
    return py::cast(RNN::load_stream(filestream));
  }

  throw std::invalid_argument("Found an invalid header byte in the saved file");
}
}  // namespace thirdai::automl::python