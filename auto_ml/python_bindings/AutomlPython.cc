#include "AutomlPython.h"
#include "AutomlDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/embedding_prototype/StringEncoder.h>
#include <auto_ml/src/udt/UDT.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <limits>

namespace thirdai::automl::python {

class ValidationOptions {
 public:
  ValidationOptions(std::string filename, std::vector<std::string> metrics,
                    std::optional<uint32_t> interval, bool use_sparse_inference)
      : _filename(std::move(filename)),
        _args(std::move(metrics), interval, use_sparse_inference) {}

  const std::string& filename() const { return _filename; }

  const udt::ValidationArgs& args() const { return _args; }

 private:
  std::string _filename;
  udt::ValidationArgs _args;
};

void defineAutomlInModule(py::module_& module) {
  py::class_<ValidationOptions>(module, "Validation")
      .def(py::init<std::string, std::vector<std::string>,
                    std::optional<uint32_t>, bool>(),
           py::arg("filename"), py::arg("metrics"),
           py::arg("interval") = std::nullopt,
           py::arg("use_sparse_inference") = false, docs::VALIDATION)
      .def("filename", &ValidationOptions::filename)
      .def("args", &ValidationOptions::args);

  py::class_<udt::ValidationArgs>(module, "ValidationArgs")
      .def_property_readonly(
          "metrics", [](udt::ValidationArgs const& v) { return v.metrics(); })
      .def_property_readonly(
          "steps_per_validation",
          [](udt::ValidationArgs const& v) { return v.stepsPerValidation(); })
      .def_property_readonly(
          "sparse_inference",
          [](udt::ValidationArgs const& v) { return v.sparseInference(); });

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
      .def("__new__", &UDTFactory::buildUDT, py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               data::UserProvidedTemporalRelationships(),
           py::arg("target"), py::arg("n_target_classes") = std::nullopt,
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("model_config") = std::nullopt,
           py::arg("options") = py::dict(), docs::UDT_INIT,
           bolt::python::OutputRedirect())
      .def("__new__", &UDTFactory::createUDTSpecifiedFileFormat,
           py::arg("file_format"), py::arg("n_target_classes"),
           py::arg("input_dim"), py::arg("model_config") = std::nullopt,
           py::arg("options") = py::dict())
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

  py::class_<udt::UDT, std::shared_ptr<udt::UDT>>(module, "UDT")
      .def("train", &udt::UDT::train, py::arg("data"), py::arg("learning_rate"),
           py::arg("epochs"), py::arg("validation") = std::nullopt,
           py::arg("batch_size") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt,
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::CallbackPtr>{},
           py::arg("verbose") = true,
           py::arg("logging_interval") = std::nullopt,
           bolt::python::OutputRedirect())
      .def("train_batch", &udt::UDT::trainBatch, py::arg("batch"),
           py::arg("learning_rate") = 0.001,
           py::arg("metrics") = std::vector<std::string>{},
           bolt::python::OutputRedirect())
      .def("evaluate", &udt::UDT::evaluate, py::arg("data"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("sparse_inference") = false,
           py::arg("return_predicted_class") = false, py::arg("verbose") = true,
           py::arg("return_metrics") = false, bolt::python::OutputRedirect())
      .def("predict", &udt::UDT::predict, py::arg("sample"),
           py::arg("sparse_inference") = false,
           py::arg("return_predicted_class") = false)
      .def("predict_batch", &udt::UDT::predictBatch, py::arg("samples"),
           py::arg("sparse_inference") = false,
           py::arg("return_predicted_class") = false)
      .def("cold_start", &udt::UDT::coldstart, py::arg("data"),
           py::arg("strong_column_names"), py::arg("weak_column_names"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("metrics"),
           py::arg("validation"), py::arg("callbacks"),
           py::arg("max_in_memory_batches") = std::nullopt, py::arg("verbose"),
           bolt::python::OutputRedirect())
      .def("embedding_representation", &udt::UDT::embedding,
           py::arg("input_sample"))
      .def("get_entity_embedding", &udt::UDT::entityEmbedding,
           py::arg("label_id"))
      .def("index", &udt::UDT::updateTemporalTrackers, py::arg("input_sample"))
      .def("index_batch", &udt::UDT::updateTemporalTrackersBatch,
           py::arg("input_samples"))
      .def("index_nodes", &udt::UDT::indexNodes, py::arg("data_source"))
      .def("clear_graph", &udt::UDT::clearGraph)
      .def("set_decode_params", &udt::UDT::setDecodeParams,
           py::arg("min_num_eval_results"),
           py::arg("top_k_per_eval_aggregation"))
      .def("reset_temporal_trackers", &udt::UDT::resetTemporalTrackers)
      .def("index_metadata", &udt::UDT::updateMetadata, py::arg("column_name"),
           py::arg("update"))
      .def("index_metadata_batch", &udt::UDT::updateMetadataBatch,
           py::arg("column_name"), py::arg("updates"))
      .def("explain", &udt::UDT::explain, py::arg("input_sample"),
           py::arg("target_class") = std::nullopt)
      .def("class_name", &udt::UDT::className)
      .def("get_data_processor", &udt::UDT::tabularDatasetFactory)
      .def("_get_model", &udt::UDT::model)
      .def("_set_model", &udt::UDT::setModel, py::arg("trained_model"))
      .def("verify_can_distribute", &udt::UDT::verifyCanDistribute)
      .def("get_encoder", &udt::UDT::getEncoder)
      .def("get_cold_start_meta_data", &udt::UDT::getColdStartMetaData)
      .def("save", &UDTFactory::save_udt, py::arg("filename"))
      .def("checkpoint", &UDTFactory::checkpoint_udt, py::arg("filename"))
      .def_static("load", &udt::UDT::load, py::arg("filename"))
      .def(bolt::python::getPickleFunction<udt::UDT>());

  py::class_<udt::StringEncoder, udt::StringEncoderPtr>(module, "StringEncoder")
      .def("supervised_train", &udt::StringEncoder::supervisedTrain,
           py::arg("data_source"), py::arg("input_col_1"),
           py::arg("input_col_2"), py::arg("label_col"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("metrics"),
           bolt::python::OutputRedirect())
      .def("encode", &udt::StringEncoder::encode, py::arg("string"))
      .def("encode_batch", &udt::StringEncoder::encodeBatch,
           py::arg("strings"));
}

void createModelsSubmodule(py::module_& module) {
  auto models_submodule = module.def_submodule("models");

  py::class_<data::TabularDatasetFactory, data::TabularDatasetFactoryPtr>(
      models_submodule, "TabularDatasetFactory")
      .def("get_dataset_loader", &data::TabularDatasetFactory::getDatasetLoader,
           py::arg("data_source"), py::arg("training"))
      .def(bolt::python::getPickleFunction<data::TabularDatasetFactory>());

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
}

void createDistributedPreprocessingWrapper(py::module_& module) {
  auto distributed_preprocessing_submodule =
      module.def_submodule("distributed_preprocessing");
  distributed_preprocessing_submodule.def(
      "preprocess_cold_start_train_source",
      &cold_start::preprocessColdStartTrainSource, py::arg("data"),
      py::arg("strong_column_names"), py::arg("weak_column_names"),
      py::arg("dataset_factory"), py::arg("metadata"));

  py::class_<cold_start::ColdStartMetaData, cold_start::ColdStartMetaDataPtr>(
      distributed_preprocessing_submodule, "ColdStartMetaData")
      .def(bolt::python::getPickleFunction<cold_start::ColdStartMetaData>());
  ;
}

void createUDTTypesSubmodule(py::module_& module) {
  auto udt_types_submodule = module.def_submodule("types");

  py::class_<automl::data::DataType,
             automl::data::DataTypePtr>(  // NOLINT
      udt_types_submodule, "ColumnType", "Base class for bolt types.")
      .def("__str__", &automl::data::DataType::toString)
      .def("__repr__", &automl::data::DataType::toString);

  // TODO(Josh): Add docs here and elsewhere
  py::class_<automl::data::NeighborsDataType, automl::data::DataType,
             automl::data::NeighborsDataTypePtr>(udt_types_submodule,
                                                 "neighbors")
      .def(py::init<>());

  py::class_<automl::data::NodeIDDataType, automl::data::DataType,
             automl::data::NodeIDDataTypePtr>(udt_types_submodule, "node_id")
      .def(py::init<>());

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
      .def(py::init<std::optional<double>, std::string>(),
           py::arg("average_n_words") = std::nullopt,
           py::arg("contextual_encoding") = "none", docs::UDT_TEXT_TYPE);

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

std::shared_ptr<udt::UDT> UDTFactory::buildUDT(
    py::object& obj, data::ColumnDataTypes data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const py::dict& options) {
  (void)obj;
  return std::make_shared<udt::UDT>(
      /* data_types = */ std::move(data_types),
      /* temporal_tracking_relationships = */
      temporal_tracking_relationships,
      /* target_col = */ target_col,
      /* n_target_classes = */ n_target_classes,
      /* integer_target = */ integer_target,
      /* time_granularity = */ std::move(time_granularity),
      /* lookahead = */ lookahead, /* delimiter = */ delimiter,
      /* model_config= */ model_config,
      /* options = */ createArgumentMap(options));
}

std::shared_ptr<udt::UDT> UDTFactory::createUDTSpecifiedFileFormat(
    py::object& obj, const std::string& file_format, uint32_t n_target_classes,
    uint32_t input_dim, const std::optional<std::string>& model_config,
    const py::dict& user_args) {
  (void)obj;
  return std::make_shared<udt::UDT>(file_format, n_target_classes, input_dim,
                                    model_config, createArgumentMap(user_args));
}

void UDTFactory::save_udt(const udt::UDT& classifier,
                          const std::string& filename) {
  classifier.model()->saveWithOptimizer(false);
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(reinterpret_cast<const char*>(&UDT_IDENTIFIER), 1);
  classifier.save_stream(filestream);
}

void UDTFactory::checkpoint_udt(const udt::UDT& classifier,
                                const std::string& filename) {
  classifier.model()->saveWithOptimizer(true);
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  filestream.write(reinterpret_cast<const char*>(&UDT_IDENTIFIER), 1);
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

py::object UDTFactory::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  uint8_t first_byte;
  filestream.read(reinterpret_cast<char*>(&first_byte), 1);

  if (first_byte == UDT_GENERATOR_IDENTIFIER) {
    return py::cast(QueryCandidateGenerator::load_stream(filestream));
  }

  if (first_byte == UDT_IDENTIFIER) {
    return py::cast(udt::UDT::load_stream(filestream));
  }

  if (first_byte == UDT_TEXT_CLASSIFIER_IDENTIFIER) {
    return py::cast(TextClassifier::load_stream(filestream));
  }

  throw std::invalid_argument("Found an invalid header byte in the saved file");
}
}  // namespace thirdai::automl::python