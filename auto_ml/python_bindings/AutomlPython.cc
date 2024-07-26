#include "AutomlPython.h"
#include "AutomlDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/udt/UDT.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::python {

template <typename T>
using NumpyArray = thirdai::bolt::python::NumpyArray<T>;

class ValidationOptions {
 public:
  ValidationOptions(std::string filename, std::vector<std::string> metrics,
                    std::optional<uint32_t> interval, bool use_sparse_inference)
      : _filename(std::move(filename)),
        _metrics(std::move(metrics)),
        _steps_per_validation(interval),
        _sparse_validation(use_sparse_inference) {}

  const std::string& filename() const { return _filename; }

  const auto& metrics() const { return _metrics; }

  auto stepsPerValidation() const { return _steps_per_validation; }

  bool sparseValidation() const { return _sparse_validation; }

 private:
  std::string _filename;
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _steps_per_validation;
  bool _sparse_validation;
};

std::shared_ptr<udt::UDT> makeUDT(
    ColumnDataTypes data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::optional<std::string>& target_col, char delimiter,
    const std::optional<std::string>& model_config,
    const py::object& pretrained_model, const py::kwargs& kwargs);

void defineAutomlInModule(py::module_& module) {
  py::class_<ValidationOptions>(module, "Validation")
      .def(py::init<std::string, std::vector<std::string>,
                    std::optional<uint32_t>, bool>(),
           py::arg("filename"), py::arg("metrics"),
           py::arg("interval") = std::nullopt,
           py::arg("use_sparse_inference") = false, docs::VALIDATION)
      .def_property_readonly("filename", &ValidationOptions::filename)
      .def_property_readonly("metrics", &ValidationOptions::metrics)
      .def_property_readonly("steps_per_validation",
                             &ValidationOptions::stepsPerValidation)
      .def_property_readonly("sparse_validation",
                             &ValidationOptions::sparseValidation);

  py::class_<udt::TrainOptions>(module, "TrainOptions")
      .def(py::init<>())
      .def_readwrite("batch_size", &udt::TrainOptions::batch_size)
      .def_readwrite("max_in_memory_batches",
                     &udt::TrainOptions::max_in_memory_batches)
      .def_readwrite("steps_per_validation",
                     &udt::TrainOptions::steps_per_validation)
      .def_readwrite("sparse_validation", &udt::TrainOptions::sparse_validation)
      .def_readwrite("verbose", &udt::TrainOptions::verbose)
      .def_readwrite("logging_interval", &udt::TrainOptions::logging_interval)
      .def_readwrite("shuffle_config", &udt::TrainOptions::shuffle_config);

  py::class_<TextDatasetConfig>(module, "TextDatasetConfig")
      .def_property_readonly("text_column", &TextDatasetConfig::textColumn)
      .def_property_readonly("label_column", &TextDatasetConfig::labelColumn)
      .def_property_readonly("label_delimiter",
                             &TextDatasetConfig::labelDelimiter);

  py::class_<udt::UDT, std::shared_ptr<udt::UDT>>(module,
                                                  "UniversalDeepTransformer")
      .def(py::init(&makeUDT), py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               UserProvidedTemporalRelationships(),
           py::arg("target") = std::nullopt, py::arg("delimiter") = ',',
           py::arg("model_config") = std::nullopt,
           py::arg("pretrained_model") = py::none(), docs::UDT_INIT,
           bolt::python::OutputRedirect())
      .def("train", &udt::UDT::train, py::arg("data"), py::arg("learning_rate"),
           py::arg("epochs"),
           py::arg("train_metrics") = std::vector<std::string>{},
           py::arg("val_data") = nullptr,
           py::arg("val_metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<udt::CallbackPtr>{},
           py::arg("options") = udt::TrainOptions(), py::arg("comm") = nullptr,
           bolt::python::OutputRedirect())
      .def("train_batch", &udt::UDT::trainBatch, py::arg("batch"),
           py::arg("learning_rate") = 0.001, bolt::python::OutputRedirect(),
           docs::UDT_TRAIN_BATCH)
      .def("set_output_sparsity", &udt::UDT::setOutputSparsity,
           py::arg("sparsity"), py::arg("rebuild_hash_tables") = false,
           bolt::python::OutputRedirect())
      .def("evaluate", &udt::UDT::evaluate, py::arg("data"),
           py::arg("metrics") = std::vector<std::string>{},
           py::arg("sparse_inference") = false, py::arg("verbose") = true,
           bolt::python::OutputRedirect())
      .def("predict", &udt::UDT::predict, py::arg("sample"),
           py::arg("sparse_inference") = false,
           py::arg("return_predicted_class") = false,
           py::arg("top_k") = std::nullopt, docs::UDT_PREDICT)
      .def("predict_batch", &udt::UDT::predictBatch, py::arg("samples"),
           py::arg("sparse_inference") = false,
           py::arg("return_predicted_class") = false,
           py::arg("top_k") = std::nullopt, docs::UDT_PREDICT_BATCH)
      .def("predict_activations_batch", &udt::UDT::predictActivationsBatch,
           py::arg("samples"), py::arg("sparse_inference") = false)
      .def("score_batch", &udt::UDT::scoreBatch, py::arg("samples"),
           py::arg("classes"), py::arg("top_k") = std::nullopt)
      .def("cold_start", &udt::UDT::coldstart, py::arg("data"),
           py::arg("strong_column_names"), py::arg("weak_column_names"),
           py::arg("variable_length") = data::VariableLengthConfig(),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("train_metrics"), py::arg("val_data"),
           py::arg("val_metrics"), py::arg("callbacks"), py::arg("options"),
           py::arg("comm") = nullptr, bolt::python::OutputRedirect())
      .def("output_correctness", &udt::UDT::outputCorrectness,
           py::arg("samples"), py::arg("labels"),
           py::arg("sparse_inference") = false,
           py::arg("num_hashes") = std::nullopt)
      .def("embedding_representation", &udt::UDT::embedding,
           py::arg("input_sample"), docs::UDT_EMBEDDING_REPRESENTATION)
      .def("get_entity_embedding", &udt::UDT::entityEmbedding,
           py::arg("label_id"), docs::UDT_ENTITY_EMBEDDING)
      .def("index", &udt::UDT::updateTemporalTrackers, py::arg("input_sample"),
           docs::UDT_INDEX)
      .def("index_batch", &udt::UDT::updateTemporalTrackersBatch,
           py::arg("input_samples"), docs::UDT_INDEX_BATCH)
      .def("reset_temporal_trackers", &udt::UDT::resetTemporalTrackers,
           docs::UDT_RESET_TEMPORAL_TRACKERS)
      .def("index_nodes", &udt::UDT::indexNodes, py::arg("data_source"),
           docs::UDT_INDEX_NODES)
      .def("clear_graph", &udt::UDT::clearGraph, docs::UDT_CLEAR_GRAPH)
      .def("set_decode_params", &udt::UDT::setDecodeParams,
           py::arg("top_k_to_return"), py::arg("num_buckets_to_eval"))
      .def("insert_new_doc_ids", &udt::UDT::insertNewDocIds, py::arg("data"))
      .def("introduce_documents", &udt::UDT::introduceDocuments,
           py::arg("data_source"), py::arg("strong_column_names"),
           py::arg("weak_column_names"),
           py::arg("num_buckets_to_sample") = std::nullopt,
           py::arg("num_random_hashes") = 0, py::arg("load_balancing") = false,
           py::arg("fast_approximation") = false, py::arg("verbose") = true,
           py::arg("sort_random_hashes") = false)
      .def("introduce_document", &udt::UDT::introduceDocument,
           py::arg("document"), py::arg("strong_column_names"),
           py::arg("weak_column_names"), py::arg("label"),
           py::arg("num_buckets_to_sample") = std::nullopt,
           py::arg("num_random_hashes") = 0, py::arg("load_balancing") = false,
           py::arg("sort_random_hashes") = false)
      .def("introduce_label", &udt::UDT::introduceLabel, py::arg("input_batch"),
           py::arg("label"), py::arg("num_buckets_to_sample") = std::nullopt,
           py::arg("num_random_hashes") = 0, py::arg("load_balancing") = false,
           py::arg("sort_random_hashes") = false)
      .def("forget", &udt::UDT::forget, py::arg("label"))
      .def("clear_index", &udt::UDT::clearIndex)
      .def("train_with_hashes", &udt::UDT::trainWithHashes, py::arg("batch"),
           py::arg("learning_rate") = 0.001)
      .def("predict_hashes", &udt::UDT::predictHashes, py::arg("sample"),
           py::arg("sparse_inference") = false,
           py::arg("force_non_empty") = true,
           py::arg("num_hashes") = std::nullopt)
      .def("predict_hashes_batch", &udt::UDT::predictHashesBatch,
           py::arg("samples"), py::arg("sparse_inference") = false,
           py::arg("force_non_empty") = true,
           py::arg("num_hashes") = std::nullopt)
      .def("associate", &udt::UDT::associate, py::arg("source_target_samples"),
           py::arg("n_buckets"),
           py::arg("n_association_samples") =
               udt::defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") =
               udt::defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = udt::defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = udt::defaults::RLHF_EPOCHS,
           py::arg("force_non_empty") = true,
           py::arg("batch_size") = udt::defaults::RLHF_BATCH_SIZE)
      .def("upvote", &udt::UDT::upvote, py::arg("source_target_samples"),
           py::arg("n_upvote_samples") = udt::defaults::RLHF_N_FEEDBACK_SAMPLES,
           py::arg("n_balancing_samples") =
               udt::defaults::RLHF_N_BALANCING_SAMPLES,
           py::arg("learning_rate") = udt::defaults::RLHF_LEARNING_RATE,
           py::arg("epochs") = udt::defaults::RLHF_EPOCHS,
           py::arg("batch_size") = udt::defaults::RLHF_BATCH_SIZE)
      .def("associate_train_data_source", &udt::UDT::associateTrain,
           py::arg("balancing_data"), py::arg("source_target_samples"),
           py::arg("n_buckets"), py::arg("n_association_samples"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("metrics"),
           py::arg("options"))
      .def("associate_cold_start_data_source", &udt::UDT::associateColdStart,
           py::arg("balancing_data"), py::arg("strong_column_names"),
           py::arg("weak_column_names"), py::arg("source_target_samples"),
           py::arg("n_buckets"), py::arg("n_association_samples"),
           py::arg("learning_rate"), py::arg("epochs"), py::arg("metrics"),
           py::arg("options"))
      .def("cold_start_with_balancing_samples",
           &udt::UDT::coldStartWithBalancingSamples, py::arg("data"),
           py::arg("strong_column_names"), py::arg("weak_column_names"),
           py::arg("learning_rate"), py::arg("epochs"),
           py::arg("train_metrics") = std::vector<std::string>{},
           py::arg("callbacks") = std::vector<bolt::callbacks::CallbackPtr>{},
           py::arg("batch_size") = std::nullopt, py::arg("verbose") = true,
           py::arg("variable_length") = data::VariableLengthConfig())
      .def("enable_rlhf", &udt::UDT::enableRlhf,
           py::arg("num_balancing_docs") = udt::defaults::MAX_BALANCING_DOCS,
           py::arg("num_balancing_samples_per_doc") =
               udt::defaults::MAX_BALANCING_SAMPLES_PER_DOC)
      .def("get_index", &udt::UDT::getIndex)
      .def("set_index", &udt::UDT::setIndex, py::arg("index"))
      .def("set_mach_sampling_threshold", &udt::UDT::setMachSamplingThreshold)
      .def("explain", &udt::UDT::explain, py::arg("input_sample"),
           py::arg("target_class") = std::nullopt, docs::UDT_EXPLAIN)
      .def("class_name", &udt::UDT::className, docs::UDT_CLASS_NAME)
      .def("_get_model", &udt::UDT::model)
      .def("_set_model", &udt::UDT::setModel, py::arg("trained_model"))
      .def("model_dims", &udt::UDT::modelDims)
      .def("text_dataset_config", &udt::UDT::textDatasetConfig)
      .def("verify_can_distribute", &udt::UDT::verifyCanDistribute)
      .def(
          "save",
          [](const std::shared_ptr<udt::UDT>& udt,
             const std::string& filename) { udt->save(filename); },
          py::arg("filename"), docs::UDT_SAVE_CHECKPOINT)
      .def("checkpoint", &udt::UDT::checkpoint, py::arg("filename"),
           docs::UDT_SAVE_CHECKPOINT)
      .def_static(
          "load",
          [](const std::string& filename) { return udt::UDT::load(filename); },
          py::arg("filename"), docs::UDT_LOAD)
      .def("get_parameters",
           [](udt::UDT& udt) {
             return thirdai::bolt::python::getParameters(udt.model());
           })
      .def("set_parameters",
           [](udt::UDT& udt, NumpyArray<float>& new_parameters) {
             thirdai::bolt::python::setParameters(udt.model(), new_parameters);
           })
      .def("is_v1", &udt::UDT::isV1)
      .def("migrate_to_v2", &udt::UDT::migrateToMachV2)
      .def(bolt::python::getPickleFunction<udt::UDT>())
      .def("save_cpp_classifier", &udt::UDT::saveCppClassifier,
           py::arg("save_path"))
      .def_static("parallel_inference", &udt::UDT::parallelInference,
                  py::arg("models"), py::arg("batch"),
                  py::arg("sparse_inference") = false,
                  py::arg("top_k") = std::nullopt)
      .def_static("label_probe_multiple_shards",
                  &udt::UDT::labelProbeMultipleShards, py::arg("shards"),
                  py::arg("batch"), py::arg("sparse_inference") = false,
                  py::arg("top_k") = std::nullopt)
      .def_static("label_probe_multiple_mach",
                  &udt::UDT::labelProbeMultipleMach, py::arg("models"),
                  py::arg("batch"), py::arg("sparse_inference") = false,
                  py::arg("top_k") = std::nullopt)
      .def_static("estimate_hash_table_size", &udt::UDT::estimateHashTableSize,
                  py::arg("output_dim"), py::arg("sparsity") = std::nullopt)
      .def("add_ner_rule", &udt::UDT::addNerRule, py::arg("rule"))
      .def("add_new_entity_to_model", &udt::UDT::addNewEntityToModel,
           py::arg("entity"));
}

void createUDTTypesSubmodule(py::module_& module) {
  auto udt_types_submodule = module.def_submodule("types");

  py::class_<DataType,
             DataTypePtr>(  // NOLINT
      udt_types_submodule, "ColumnType", "Base class for bolt types.")
      .def("__str__", &DataType::toString)
      .def("__repr__", &DataType::toString);

  // TODO(Any): Add docs for graph UDT types
  py::class_<NeighborsDataType, DataType, NeighborsDataTypePtr>(
      udt_types_submodule, "neighbors")
      .def(py::init<>());

  py::class_<NodeIDDataType, DataType, NodeIDDataTypePtr>(udt_types_submodule,
                                                          "node_id")
      .def(py::init<>());

  py::class_<CategoricalMetadataConfig, CategoricalMetadataConfigPtr>(
      udt_types_submodule, "metadata")
      .def(py::init<std::string, std::string, ColumnDataTypes, char>(),
           py::arg("filename"), py::arg("key_column_name"),
           py::arg("data_types"), py::arg("delimiter") = ',',
           docs::UDT_CATEGORICAL_METADATA_CONFIG);

  py::class_<CategoricalDataType, DataType, CategoricalDataTypePtr>(
      udt_types_submodule, "categorical")
      .def(py::init<std::optional<size_t>, std::string, std::optional<char>,
                    CategoricalMetadataConfigPtr>(),
           py::arg("n_classes") = std::nullopt, py::arg("type") = "str",
           py::arg("delimiter") = std::nullopt, py::arg("metadata") = nullptr,
           docs::UDT_CATEGORICAL_TYPE)
      .def_property_readonly("delimiter", [](CategoricalDataType& categorical) {
        return categorical.delimiter;
      });

  py::class_<NumericalDataType, DataType, NumericalDataTypePtr>(
      udt_types_submodule, "numerical")
      .def(py::init<std::pair<double, double>, std::string,
                    std::optional<size_t>>(),
           py::arg("range"), py::arg("granularity") = "m",
           py::arg("explicit_granularity") = std::nullopt,
           docs::UDT_NUMERICAL_TYPE);

  py::class_<TextDataType, DataType, TextDataTypePtr>(udt_types_submodule,
                                                      "text")
      // TODO(any): run benchmarks to improve the defaults
      .def(py::init<std::string, std::string, bool>(),
           py::arg("tokenizer") = "words",
           py::arg("contextual_encoding") = "none", py::arg("lowercase") = true,
           docs::UDT_TEXT_TYPE)
      .def(py::init<dataset::WordpieceTokenizerPtr, std::string>(),
           py::arg("tokenizer"), py::arg("contextual_encoding") = "none",
           docs::UDT_TEXT_TYPE);

  py::class_<DateDataType, DataType, DateDataTypePtr>(udt_types_submodule,
                                                      "date")
      .def(py::init<>(), docs::UDT_DATE_TYPE);

  py::class_<SequenceDataType, DataType, SequenceDataTypePtr>(
      udt_types_submodule, "sequence")
      .def(py::init<std::optional<size_t>, char, std::optional<uint32_t>>(),
           py::arg("n_classes") = std::nullopt, py::arg("delimiter") = ' ',
           py::arg("max_length") = std::nullopt, docs::UDT_SEQUENCE_TYPE);

  py::class_<TokenTagsDataType, DataType, TokenTagsDataTypePtr>(
      udt_types_submodule, "token_tags")
      .def(py::init<std::vector<std::string>, std::string>(), py::arg("tags"),
           py::arg("default_tag"));
}

void createUDTTemporalSubmodule(py::module_& module) {
  auto udt_temporal_submodule = module.def_submodule("temporal");

  py::class_<TemporalConfig>(  // NOLINT
      udt_temporal_submodule, "TemporalConfig",
      "Base class for temporal feature configs.");

  udt_temporal_submodule.def(
      "categorical", TemporalConfig::categorical, py::arg("column_name"),
      py::arg("track_last_n"), py::arg("column_known_during_inference") = false,
      py::arg("use_metadata") = false, docs::UDT_CATEGORICAL_TEMPORAL);

  udt_temporal_submodule.def("numerical", TemporalConfig::numerical,
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

        return config::buildModel(json_config, user_input, input_dims,
                                  /* mach= */ false);
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
    } else if (py::isinstance<py::list>(v)) {
      bool success = false;
      try {
        std::vector<int32_t> value = v.cast<std::vector<int32_t>>();
        args.insert(name, value);
        success = true;  // NOLINT (clang-tidy thinks this is unused)
      } catch (...) {
      }
      try {
        auto value = v.cast<std::vector<dataset::TextTokenizerPtr>>();
        args.insert(name, value);
        success = true;  // NOLINT (clang-tidy thinks this is unused)
      } catch (...) {
      }
      if (!success) {
        throw std::invalid_argument(
            "Invalid type for argument '" + name +
            "'. Must be either List[int] or List[dataset.Tokenizer].");
      }
    } else if (py::isinstance<data::FeatureEnhancementConfig>(v)) {
      auto value = v.cast<data::FeatureEnhancementConfig>();
      args.insert(name, value);
    } else if (v.is_none()) {
    } else {
      throw std::invalid_argument(
          "Invalid type '" + py::str(v.get_type()).cast<std::string>() +
          "'. Values of parameters dictionary must be "
          "bool, int, float, str, list of integers or UDTConfig.");
    }
  }

  return args;
}

std::shared_ptr<udt::UDT> makeUDT(
    ColumnDataTypes data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::optional<std::string>& target_col, char delimiter,
    const std::optional<std::string>& model_config,
    const py::object& pretrained_model, const py::kwargs& kwargs) {
  if (kwargs.contains("integer_target")) {
    throw std::invalid_argument(
        "Argument 'integer_target' is deprecated. Please use "
        "bolt.types.categorical(type='int'), or type='str' for the target data "
        "type.");
  }

  if (kwargs.contains("n_target_classes")) {
    throw std::invalid_argument(
        "Argument 'n_target_classes' is deprecated. Please use "
        "bolt.types.categorical(n_classes=10) for the target data type.");
  }

  if (kwargs.contains("options")) {
    throw std::invalid_argument(
        "Argument 'options' is deprecated. Please use pass any args from "
        "options as regular kwargs to UDT.");
  }

  if (!target_col.has_value()) {
    throw std::invalid_argument(
        "The 'target' parameter is required but was not specified. Please "
        "provide a valid column name.");
  }

  return std::make_shared<udt::UDT>(
      /* data_types = */ std::move(data_types),
      /* temporal_tracking_relationships = */ temporal_tracking_relationships,
      /* target_col = */ target_col.value(),
      /* delimiter = */ delimiter,
      /* model_config= */ model_config,
      /* pretrained_model= */ pretrained_model,
      /* options = */ createArgumentMap(kwargs));
}

}  // namespace thirdai::automl::python