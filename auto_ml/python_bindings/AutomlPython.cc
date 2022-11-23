#include "AutomlPython.h"
#include "AutomlDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>

namespace thirdai::automl::python {

void defineAutomlInBoltSubmodule(py::module_& bolt_submodule) {
  py::class_<models::ValidationOptions>(bolt_submodule, "Validation")
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
  py::class_<UDTFactory>(bolt_submodule, "UniversalDeepTransformer",
                         docs::UDT_CLASS)
      .def("__new__", &UDTFactory::buildUDTClassifierWrapper,
           py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               data::UserProvidedTemporalRelationships(),
           py::arg("target"), py::arg("n_target_classes") = std::nullopt,
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("model_config") = std::nullopt,
           py::arg("options") = models::OptionsMap(), docs::UDT_INIT,
           bolt::python::OutputRedirect())
      .def("__new__", &UDTFactory::buildUDTGeneratorWrapper,
           py::arg("source_column"), py::arg("target_column"),
           py::arg("dataset_size"), docs::UDT_GENERATOR_INIT)

      .def_static("load", &UDTFactory::load, py::arg("filename"),
                  docs::UDT_CLASSIFIER_AND_GENERATOR_LOAD);
}

void createModelsSubmodule(py::module_& bolt_submodule) {
  auto models_submodule = bolt_submodule.def_submodule("models");

  py::class_<ModelPipeline, std::shared_ptr<ModelPipeline>>(models_submodule,
                                                            "Pipeline")
      .def(py::init(&createPipeline), py::arg("deployment_config"),
           py::arg("parameters") = py::dict(),
           docs::MODEL_PIPELINE_INIT_FROM_CONFIG,
           bolt::python::OutputRedirect())
      .def(py::init(&createPipelineFromSavedConfig), py::arg("config_path"),
           py::arg("parameters") = py::dict(),
           docs::MODEL_PIPELINE_INIT_FROM_SAVED_CONFIG,
           bolt::python::OutputRedirect())
      .def("train_with_file", &ModelPipeline::trainOnFile, py::arg("filename"),
           py::arg("train_config"), py::arg("batch_size") = std::nullopt,
           py::arg("validation") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt,
           docs::MODEL_PIPELINE_TRAIN_FILE, bolt::python::OutputRedirect())
      .def("train_with_loader", &ModelPipeline::trainOnDataLoader,
           py::arg("data_source"), py::arg("train_config"),
           py::arg("validation") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt,
           docs::MODEL_PIPELINE_TRAIN_DATA_LOADER,
           bolt::python::OutputRedirect())
      .def("evaluate_with_file", &evaluateOnFileWrapper, py::arg("filename"),
           py::arg("eval_config") = std::nullopt,
           docs::MODEL_PIPELINE_EVALUATE_FILE, bolt::python::OutputRedirect())
      .def("evaluate_with_loader", &evaluateOnDataLoaderWrapper,
           py::arg("data_source"), py::arg("eval_config") = std::nullopt,
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
           docs::MODEL_PIPELINE_GET_DATA_PROCESSOR)
      .def_property_readonly("default_train_batch_size",
                             &ModelPipeline::defaultBatchSize)
      .def_property_readonly_static(
          "default_evaluate_batch_size", [](const py::object& /* self */) {
            return models::DEFAULT_EVALUATE_BATCH_SIZE;
          });

  py::class_<data::UDTDatasetFactory, data::UDTDatasetFactoryPtr>(
      models_submodule, "TemporalContext")
      .def("reset", &data::UDTDatasetFactory::resetTemporalTrackers,
           docs::TEMPORAL_CONTEXT_RESET)
      .def("update_temporal_trackers",
           py::overload_cast<const LineInput&>(
               &data::UDTDatasetFactory::updateTemporalTrackers),
           py::arg("update"), docs::TEMPORAL_CONTEXT_UPDATE)
      .def("batch_update_temporal_trackers",
           py::overload_cast<const LineInputBatch&>(
               &data::UDTDatasetFactory::batchUpdateTemporalTrackers),
           py::arg("updates"), docs::TEMPORAL_CONTEXT_UPDATE_BATCH);

  py::class_<data::UDTConfig, data::UDTConfigPtr>(models_submodule, "UDTConfig")
      .def(py::init<data::ColumnDataTypes,
                    data::UserProvidedTemporalRelationships, std::string,
                    uint32_t, bool, std::string, uint32_t, char>(),
           py::arg("data_types"), py::arg("temporal_tracking_relationships"),
           py::arg("target"), py::arg("n_target_classes"),
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', docs::ORACLE_CONFIG_INIT,
           bolt::python::OutputRedirect());

  py::class_<UniversalDeepTransformer, ModelPipeline,
             std::shared_ptr<UniversalDeepTransformer>>(models_submodule,
                                                        "UDTClassifier")
      .def(py::init(&UniversalDeepTransformer::buildUDT), py::arg("data_types"),
           py::arg("temporal_tracking_relationships") =
               data::UserProvidedTemporalRelationships(),
           py::arg("target"), py::arg("n_target_classes") = std::nullopt,
           py::arg("integer_target") = false,
           py::arg("time_granularity") = "daily", py::arg("lookahead") = 0,
           py::arg("delimiter") = ',', py::arg("model_config") = std::nullopt,
           py::arg("options") = models::OptionsMap(), docs::UDT_INIT,
           bolt::python::OutputRedirect())
      .def("class_name", &UniversalDeepTransformer::className,
           py::arg("neuron_id"), docs::UDT_CLASS_NAME)
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

  py::class_<QueryCandidateGenerator, std::shared_ptr<QueryCandidateGenerator>>(
      models_submodule, "UDTGenerator")
      .def(py::init(&QueryCandidateGenerator::buildGeneratorFromDefaultConfig),
           py::arg("source_column"), py::arg("target_column"),
           py::arg("dataset_size"), docs::UDT_GENERATOR_INIT)
      .def("train", &QueryCandidateGenerator::buildFlashIndex,
           py::arg("filename"), docs::UDT_GENERATOR_TRAIN)
      .def("evaluate", &QueryCandidateGenerator::evaluateOnFile,
           py::arg("filename"), py::arg("top_k"), docs::UDT_GENERATOR_EVALUATE)
      .def(
          "predict",
          [](QueryCandidateGenerator& udt_generator_model,
             const std::string& sample, uint32_t top_k) {
            return udt_generator_model.queryFromList({sample}, top_k);
          },
          py::arg("query"), py::arg("top_k"), docs::UDT_GENERATOR_PREDICT)
      .def("predict_batch", &QueryCandidateGenerator::queryFromList,
           py::arg("queries"), py::arg("top_k"),
           docs::UDT_GENERATOR_PREDICT_BATCH)
      .def("save", &UDTFactory::save_generator, py::arg("filename"),
           docs::UDT_GENERATOR_SAVE);
}

ModelPipeline createPipeline(const deployment::DeploymentConfigPtr& config,
                             const py::dict& parameters) {
  deployment::UserInputMap cpp_parameters;
  for (const auto& [k, v] : parameters) {
    if (!py::isinstance<py::str>(k)) {
      throw std::invalid_argument("Keys of parameters map must be strings.");
    }
    std::string name = k.cast<std::string>();

    if (py::isinstance<py::bool_>(v)) {
      bool value = v.cast<bool>();
      cpp_parameters.emplace(name, deployment::UserParameterInput(value));
    } else if (py::isinstance<py::int_>(v)) {
      uint32_t value = v.cast<uint32_t>();
      cpp_parameters.emplace(name, deployment::UserParameterInput(value));
    } else if (py::isinstance<py::float_>(v)) {
      float value = v.cast<float>();
      cpp_parameters.emplace(name, deployment::UserParameterInput(value));
    } else if (py::isinstance<py::str>(v)) {
      std::string value = v.cast<std::string>();
      cpp_parameters.emplace(name, deployment::UserParameterInput(value));
    } else if (py::isinstance<data::UDTConfig>(v)) {
      data::UDTConfigPtr value = v.cast<data::UDTConfigPtr>();
      cpp_parameters.emplace(name, deployment::UserParameterInput(value));
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
  auto config = deployment::DeploymentConfig::load(config_path);

  return createPipeline(config, parameters);
}

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::EvalConfig>& eval_config) {
  auto output = model.evaluate(data_source, eval_config);

  return convertInferenceTrackerToNumpy(output);
}

py::object evaluateOnFileWrapper(ModelPipeline& model,
                                 const std::string& filename,
                                 std::optional<bolt::EvalConfig>& eval_config) {
  return evaluateOnDataLoaderWrapper(
      model,
      dataset::SimpleFileDataLoader::make(filename,
                                          models::DEFAULT_EVALUATE_BATCH_SIZE),
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

// UDT Factory Methods

QueryCandidateGenerator UDTFactory::buildUDTGeneratorWrapper(
    py::object& obj, const std::string& source_column,
    const std::string& target_column, const std::string& dataset_size) {
  (void)obj;
  return QueryCandidateGenerator::buildGeneratorFromDefaultConfig(
      /* source_column_name = */ source_column,
      /* target_column_name = */ target_column,
      /* dataset_size = */ dataset_size);
}

UniversalDeepTransformer UDTFactory::buildUDTClassifierWrapper(
    py::object& obj, data::ColumnDataTypes data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    std::string target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const std::unordered_map<std::string, std::string>& options) {
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
      /* model_config= */ model_config,
      /* options = */ options);
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

  throw std::invalid_argument("Found an invalid header byte in the saved file");
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

}  // namespace thirdai::automl::python