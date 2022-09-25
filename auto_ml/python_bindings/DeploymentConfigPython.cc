#include "DeploymentConfigPython.h"
#include <bolt/python_bindings/ConversionUtils.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/BlockConfig.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/ModelConfig.h>
#include <auto_ml/src/deployment_config/NodeConfig.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <auto_ml/src/deployment_config/dataset_configs/BasicClassificationDataset.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::automl::deployment_config::python {

void createDeploymentConfigSubmodule(py::module_& thirdai_module) {
  py::module_ submodule = thirdai_module.def_submodule("deployment_config");

  py::class_<HyperParameter<uint32_t>, HyperParameterPtr<uint32_t>>(  // NOLINT
      submodule, "UintHyperParameter");

  py::class_<HyperParameter<float>, HyperParameterPtr<float>>(  // NOLINT
      submodule, "FloatHyperParameter");

  py::class_<HyperParameter<std::string>,  // NOLINT
             HyperParameterPtr<std::string>>(submodule, "StrHyperParameter");

  py::class_<HyperParameter<bool>, HyperParameterPtr<bool>>(  // NOLINT
      submodule, "BoolHyperParameter");

  py::class_<HyperParameter<std::shared_ptr<bolt::LossFunction>>,  // NOLINT
             HyperParameterPtr<std::shared_ptr<bolt::LossFunction>>>(
      submodule, "LossHyperParameter");

  py::class_<HyperParameter<bolt::SamplingConfigPtr>,  // NOLINT
             HyperParameterPtr<bolt::SamplingConfigPtr>>(
      submodule, "SamplingConfigHyperParameter");

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
  defConstantParameter<std::shared_ptr<bolt::LossFunction>>(submodule);
  defConstantParameter<bolt::SamplingConfigPtr>(submodule);

  defOptionParameter<bool>(submodule);
  defOptionParameter<uint32_t>(submodule);
  defOptionParameter<float>(submodule);
  defOptionParameter<std::string>(submodule);
  defOptionParameter<std::shared_ptr<bolt::LossFunction>>(submodule);
  defOptionParameter<bolt::SamplingConfigPtr>(submodule);

  submodule.def("UserSpecifiedParameter", &makeUserSpecifiedParameter,
                py::arg("name"), py::arg("type"));

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
          py::arg("sampling_config") = std::nullopt);

  py::class_<ModelConfig, ModelConfigPtr>(submodule, "ModelConfig")
      .def(py::init<std::vector<std::string>, std::vector<NodeConfigPtr>,
                    HyperParameterPtr<std::shared_ptr<bolt::LossFunction>>>(),
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
           py::arg("use_pairgrams"), py::arg("range"));

  py::class_<DatasetConfig, DatasetConfigPtr>(submodule,  // NOLINT
                                              "DatasetConfig");

  py::class_<BasicClassificationDatasetConfig, DatasetConfig,
             std::shared_ptr<BasicClassificationDatasetConfig>>(
      submodule, "BasicClassificationDatasetConfig")
      .def(py::init<BlockConfigPtr, BlockConfigPtr, HyperParameterPtr<bool>,
                    HyperParameterPtr<std::string>>(),
           py::arg("data_block"), py::arg("label_block"), py::arg("shuffle"),
           py::arg("delimiter"));

  py::class_<TrainEvalParameters>(submodule, "TrainTestParameters")
      .def(py::init<std::optional<uint32_t>, std::optional<uint32_t>, uint32_t,
                    bool, std::vector<std::string>>(),
           py::arg("rebuild_hash_tables_interval"),
           py::arg("reconstruct_hash_functions_interval"),
           py::arg("default_batch_size"), py::arg("use_sparse_inference"),
           py::arg("evaluation_metrics"));

  py::class_<DeploymentConfig, DeploymentConfigPtr>(submodule,
                                                    "DeploymentConfig")
      .def(py::init<DatasetConfigPtr, ModelConfigPtr, TrainEvalParameters>(),
           py::arg("model_config"), py::arg("dataset_config"),
           py::arg("train_eval_parameters"));

  py::class_<ModelPipeline>(submodule, "ModelPipeline")
      .def(py::init<DeploymentConfigPtr, const std::string&,
                    const UserInputMap&>(),
           py::arg("deployment_config"), py::arg("size"), py::arg("parameters"))
      .def("train",
           py::overload_cast<const std::string&, uint32_t, float,
                             std::optional<uint32_t>, std::optional<uint32_t>>(
               &ModelPipeline::train),
           py::arg("filename"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("batch_size") = std::nullopt,
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("train",
           py::overload_cast<const std::shared_ptr<dataset::DataLoader>&,
                             uint32_t, float, std::optional<uint32_t>>(
               &ModelPipeline::train),
           py::arg("filename"), py::arg("epochs"), py::arg("learning_rate"),
           py::arg("max_in_memory_batches") = std::nullopt)
      .def("evaluate", &evaluateWrapperFilename, py::arg("filename"))
      .def("evaluate", &evaluateWrapperDataLoader, py::arg("data_source"));
}

template <typename T>
void defConstantParameter(py::module_& submodule) {
  submodule.def("ConstantParameter", &ConstantParameter<T>::make,
                py::arg("value").noconvert());
}

template <typename T>
void defOptionParameter(py::module_& submodule) {
  submodule.def("OptionParameter", &OptionParameter<T>::make,
                py::arg("values").noconvert());
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

  throw std::invalid_argument("Invalid type '" +
                              py::str(type).cast<std::string>() +
                              "' passed to UserSpecifiedParameter.");
}

py::object evaluateWrapperDataLoader(
    ModelPipeline& model,
    const std::shared_ptr<dataset::DataLoader>& data_source) {
  auto [metrics, output] = model.evaluate(data_source);

  return bolt::python::constructNumpyActivationsArrays(metrics, output);
}

py::object evaluateWrapperFilename(ModelPipeline& model,
                                   const std::string& filename) {
  return evaluateWrapperDataLoader(
      model, std::make_shared<dataset::SimpleFileDataLoader>(
                 filename, model.defaultBatchSize()));
}

}  // namespace thirdai::automl::deployment_config::python