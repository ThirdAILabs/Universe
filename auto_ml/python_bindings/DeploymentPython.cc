#include "DeploymentPython.h"
#include <bolt/python_bindings/ConversionUtils.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/Artifact.h>
#include <auto_ml/src/deployment_config/BlockConfig.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/ModelConfig.h>
#include <auto_ml/src/deployment_config/NodeConfig.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <auto_ml/src/deployment_config/dataset_configs/SingleBlockDatasetFactory.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/Aliases.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleDatasetFactory.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/TemporalContext.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
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

namespace py = pybind11;

namespace thirdai::automl::deployment::python {

void createDeploymentSubmodule(py::module_& thirdai_module) {
  py::module_ submodule = thirdai_module.def_submodule("deployment");

  py::class_<HyperParameter<uint32_t>, HyperParameterPtr<uint32_t>>(  // NOLINT
      submodule, "UintHyperParameter");

  py::class_<HyperParameter<float>, HyperParameterPtr<float>>(  // NOLINT
      submodule, "FloatHyperParameter");

  py::class_<HyperParameter<std::string>,  // NOLINT
             HyperParameterPtr<std::string>>(submodule, "StrHyperParameter");

  py::class_<HyperParameter<bool>, HyperParameterPtr<bool>>(  // NOLINT
      submodule, "BoolHyperParameter");

  py::class_<HyperParameter<bolt::SamplingConfigPtr>,  // NOLINT
             HyperParameterPtr<bolt::SamplingConfigPtr>>(
      submodule, "SamplingConfigHyperParameter");

  py::class_<HyperParameter<OracleConfigPtr>,  // NOLINT
             HyperParameterPtr<OracleConfigPtr>>(submodule,
                                                 "OracleConfigHyperParameter");

  /**
   * Do not change the order of these overloads. Because bool is a sublclass of
   * int in python, it must be declared first or calling this function with a
   * bool will result in the uint32_t function being called. Pybind guarentees
   * that overloads are tried in the order they were registered so this is safe
   * to do.
   */
  defConstantParameter<bool>(submodule);
  defConstantParameter<uint32_t>(submodule);
  defConstantParameter<float>(submodule);
  defConstantParameter<std::string>(submodule);
  defConstantParameter<bolt::SamplingConfigPtr>(submodule);
  defConstantParameter<OracleConfigPtr>(submodule);

  defOptionMappedParameter<bool>(submodule);
  defOptionMappedParameter<uint32_t>(submodule);
  defOptionMappedParameter<float>(submodule);
  defOptionMappedParameter<std::string>(submodule);
  defOptionMappedParameter<bolt::SamplingConfigPtr>(submodule);
  defOptionMappedParameter<OracleConfigPtr>(submodule);

  submodule.def("UserSpecifiedParameter", &makeUserSpecifiedParameter,
                py::arg("name"), py::arg("type"),
                py::arg("default_value") = py::cast(std::nullopt));

  py::class_<AutotunedSparsityParameter, HyperParameter<float>,
             std::shared_ptr<AutotunedSparsityParameter>>(
      submodule, "AutotunedSparsityParameter")
      .def(py::init<std::string>(), py::arg("dimension_param_name"));

  py::class_<DatasetLabelDimensionParameter, HyperParameter<uint32_t>,
             std::shared_ptr<DatasetLabelDimensionParameter>>(
      submodule, "DatasetLabelDimensionParameter")
      .def(py::init<>())
      // This is why we pass in a py::object:
      // https://stackoverflow.com/questions/70504125/pybind11-pyclass-def-property-readonly-static-incompatible-function-arguments
      .def_property_readonly_static("dimension_param_name", [](py::object&) {
        return DatasetLabelDimensionParameter::PARAM_NAME;
      });

  py::class_<NodeConfig, NodeConfigPtr>(submodule, "NodeConfig");  // NOLINT

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
          py::arg("sampling_config") = std::nullopt)
      .def(py::init<std::string, HyperParameterPtr<uint32_t>,
                    HyperParameterPtr<std::string>, std::string>(),
           py::arg("name"), py::arg("dim"), py::arg("activation"),
           py::arg("predecessor"));

  py::class_<ModelConfig, ModelConfigPtr>(submodule, "ModelConfig")
      .def(py::init<std::vector<std::string>, std::vector<NodeConfigPtr>,
                    std::shared_ptr<bolt::LossFunction>>(),
           py::arg("input_names"), py::arg("nodes"), py::arg("loss"));

  py::class_<BlockConfig, BlockConfigPtr>(submodule, "BlockConfig");  // NOLINT

  py::class_<NumericalCategoricalBlockConfig, BlockConfig,
             std::shared_ptr<NumericalCategoricalBlockConfig>>(
      submodule, "NumericalCategoricalBlockConfig")
      .def(py::init<HyperParameterPtr<uint32_t>,
                    HyperParameterPtr<std::string>>(),
           py::arg("n_classes"), py::arg("delimiter"));

  py::class_<DenseArrayBlockConfig, BlockConfig,
             std::shared_ptr<DenseArrayBlockConfig>>(submodule,
                                                     "DenseArrayBlockConfig")
      .def(py::init<HyperParameterPtr<uint32_t>>(), py::arg("dim"));

  py::class_<TextBlockConfig, BlockConfig, std::shared_ptr<TextBlockConfig>>(
      submodule, "TextBlockConfig")
      .def(py::init<bool, HyperParameterPtr<uint32_t>>(),
           py::arg("use_pairgrams"), py::arg("range"))
      .def(py::init<bool>(), py::arg("use_pairgrams"));

  py::class_<DatasetLoaderFactoryConfig,  // NOLINT
             DatasetLoaderFactoryConfigPtr>(submodule, "DatasetConfig");

  py::class_<SingleBlockDatasetFactoryConfig, DatasetLoaderFactoryConfig,
             std::shared_ptr<SingleBlockDatasetFactoryConfig>>(
      submodule, "SingleBlockDatasetFactory")
      .def(py::init<BlockConfigPtr, BlockConfigPtr, HyperParameterPtr<bool>,
                    HyperParameterPtr<std::string>>(),
           py::arg("data_block"), py::arg("label_block"), py::arg("shuffle"),
           py::arg("delimiter"));

  py::class_<OracleDatasetFactoryConfig, DatasetLoaderFactoryConfig,
             std::shared_ptr<OracleDatasetFactoryConfig>>(
      submodule, "OracleDatasetFactory")
      .def(py::init<HyperParameterPtr<OracleConfigPtr>, HyperParameterPtr<bool>,
                    HyperParameterPtr<uint32_t>>(),
           py::arg("config"), py::arg("parallel"),
           py::arg("text_pairgram_word_limit"));

  py::class_<TrainEvalParameters>(submodule, "TrainEvalParameters")
      .def(py::init<std::optional<uint32_t>, std::optional<uint32_t>, uint32_t,
                    bool, std::optional<float>>(),
           py::arg("rebuild_hash_tables_interval"),
           py::arg("reconstruct_hash_functions_interval"),
           py::arg("default_batch_size"), py::arg("freeze_hash_tables"),
           py::arg("prediction_threshold") = std::nullopt);

  py::class_<DeploymentConfig, DeploymentConfigPtr>(submodule,
                                                    "DeploymentConfig")
      .def(py::init<DatasetLoaderFactoryConfigPtr, ModelConfigPtr,
                    TrainEvalParameters>(),
           py::arg("dataset_config"), py::arg("model_config"),
           py::arg("train_eval_parameters"))
      .def("save", &DeploymentConfig::save, py::arg("filename"))
      .def_static("load", &DeploymentConfig::load, py::arg("filename"));

  py::class_<ModelPipeline>(submodule, "ModelPipeline")
      .def(py::init(&createPipeline), py::arg("deployment_config"),
           py::arg("parameters") = py::dict())
      .def(py::init(&createPipelineFromSavedConfig), py::arg("config_path"),
           py::arg("parameters") = py::dict())
      .def("train", &ModelPipeline::trainOnFile, py::arg("filename"),
           py::arg("train_config"), py::arg("batch_size") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("train", &ModelPipeline::trainOnDataLoader, py::arg("data_source"),
           py::arg("train_config"),
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("evaluate", &evaluateOnFileWrapper, py::arg("filename"),
           py::arg("predict_config") = std::nullopt)
      .def("evaluate", &evaluateOnDataLoaderWrapper, py::arg("data_source"),
           py::arg("predict_config") = std::nullopt)
      .def("predict", &predictWrapper, py::arg("input_sample"),
           py::arg("use_sparse_inference") = false)
      .def("predict_tokens", &predictTokensWrapper, py::arg("tokens"),
           py::arg("use_sparse_inference") = false)
      .def("predict_batch", &predictBatchWrapper, py::arg("input_samples"),
           py::arg("use_sparse_inference") = false)
      .def("load_validation_data", &ModelPipeline::loadValidationDataFromFile,
           py::arg("filename"))
      .def("save", &ModelPipeline::save, py::arg("filename"))
      // getArtifact returns a variant which then gets resolved to one of its
      // contained types.
      .def("get_artifact", &ModelPipeline::getArtifact)
      .def("list_artifact_names", &ModelPipeline::listArtifactNames)
      .def_static("load", &ModelPipeline::load, py::arg("filename"));

  py::class_<OracleConfig, OracleConfigPtr>(submodule, "OracleConfig")
      .def(py::init<ColumnDataTypes, UserProvidedTemporalRelationships,
                    std::string, std::string, uint32_t, char>(),
           py::arg("data_types"), py::arg("temporal_tracking_relationships"),
           py::arg("target"), py::arg("time_granularity") = "daily",
           py::arg("lookahead") = 0, py::arg("delimiter") = ',');

  py::class_<TemporalContext, TemporalContextPtr>(submodule, "TemporalContext")
      .def(py::init<>())
      .def("reset", &TemporalContext::reset)
      .def("update_temporal_trackers", &TemporalContext::updateTemporalTrackers,
           py::arg("update"))
      .def("batch_update_temporal_trackers",
           &TemporalContext::batchUpdateTemporalTrackers, py::arg("updates"));
}

template <typename T>
void defConstantParameter(py::module_& submodule) {
  submodule.def("ConstantParameter", &ConstantParameter<T>::make,
                py::arg("value").noconvert());
}

template <typename T>
void defOptionMappedParameter(py::module_& submodule) {
  submodule.def("OptionMappedParameter", &OptionMappedParameter<T>::make,
                py::arg("option_name"), py::arg("values").noconvert());
}

py::object makeUserSpecifiedParameter(const std::string& name,
                                      const py::object& type,
                                      const py::object& default_value) {
  if (py::str(type).cast<std::string>() == "<class 'bool'>") {
    return py::cast(UserSpecifiedParameter<bool>::make(
        name, default_value.cast<std::optional<bool>>()));
  }

  if (py::str(type).cast<std::string>() == "<class 'int'>") {
    return py::cast(UserSpecifiedParameter<uint32_t>::make(
        name, default_value.cast<std::optional<uint32_t>>()));
  }

  if (py::str(type).cast<std::string>() == "<class 'float'>") {
    return py::cast(UserSpecifiedParameter<float>::make(
        name, default_value.cast<std::optional<float>>()));
  }

  if (py::str(type).cast<std::string>() == "<class 'str'>") {
    return py::cast(UserSpecifiedParameter<std::string>::make(
        name, default_value.cast<std::optional<std::string>>()));
  }

  if (py::str(type).cast<std::string>() ==
      "<class 'thirdai._thirdai.deployment.OracleConfig'>") {
    return py::cast(UserSpecifiedParameter<OracleConfigPtr>::make(
        name, default_value.cast<std::optional<OracleConfigPtr>>()));
  }

  throw std::invalid_argument("Invalid type '" +
                              py::str(type).cast<std::string>() +
                              "' passed to UserSpecifiedParameter. Must be one "
                              "of bool, int, float, str, or OracleConfig.");
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
    } else if (py::isinstance<OracleConfig>(v)) {
      OracleConfigPtr value = v.cast<OracleConfigPtr>();
      cpp_parameters.emplace(name, UserParameterInput(value));
    } else {
      throw std::invalid_argument(
          "Invalid type '" + py::str(v.get_type()).cast<std::string>() +
          "'. Values of parameters dictionary must be "
          "bool, int, float, str, OracleConfig, or TemporalContext.");
    }
  }

  return ModelPipeline::make(config, cpp_parameters);
}

ModelPipeline createPipelineFromSavedConfig(const std::string& config_path,
                                            const py::dict& parameters) {
  auto config = DeploymentConfig::load(config_path);

  return createPipeline(config, parameters);
}

py::object evaluateOnDataLoaderWrapper(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source,
    std::optional<bolt::PredictConfig>& predict_config) {
  auto output = model.evaluate(data_source, predict_config);

  return convertInferenceTrackerToNumpy(output);
}

py::object evaluateOnFileWrapper(
    ModelPipeline& model, const std::string& filename,
    std::optional<bolt::PredictConfig>& predict_config) {
  return evaluateOnDataLoaderWrapper(model,
                                     dataset::SimpleFileDataLoader::make(
                                         filename, DEFAULT_EVALUATE_BATCH_SIZE),
                                     predict_config);
}

py::object predictWrapper(ModelPipeline& model, const std::string& sample,
                          bool use_sparse_inference) {
  BoltVector output = model.predict(sample, use_sparse_inference);
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

py::object predictBatchWrapper(ModelPipeline& model,
                               const std::vector<std::string>& samples,
                               bool use_sparse_inference) {
  BoltBatch outputs = model.predictBatch(samples, use_sparse_inference);

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