#include "DeploymentPython.h"
#include "DeploymentDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <auto_ml/src/deployment_config/BlockConfig.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/ModelConfig.h>
#include <auto_ml/src/deployment_config/NodeConfig.h>
#include <auto_ml/src/deployment_config/TrainEvalParameters.h>
#include <auto_ml/src/deployment_config/dataset_configs/SingleBlockDatasetFactoryConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/UDTDatasetFactoryConfig.h>
#include <dataset/src/utils/TextEncodingUtils.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>

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

  py::class_<HyperParameter<data::UDTConfigPtr>,  // NOLINT
             HyperParameterPtr<data::UDTConfigPtr>>(submodule,
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
  defConstantParameter<data::UDTConfigPtr>(submodule,
                                           /* add_docs= */ false);

  defOptionMappedParameter<bool>(submodule, /* add_docs= */ true);
  defOptionMappedParameter<uint32_t>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<float>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<std::string>(submodule, /* add_docs= */ false);
  defOptionMappedParameter<bolt::SamplingConfigPtr>(submodule,
                                                    /* add_docs= */ false);
  defOptionMappedParameter<data::UDTConfigPtr>(submodule,
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
           docs::MODEL_CONFIG_INIT)
      .def("save", &ModelConfig::save, py::arg("filename"));

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
      .def(py::init<HyperParameterPtr<data::UDTConfigPtr>,
                    HyperParameterPtr<bool>, HyperParameterPtr<uint32_t>,
                    HyperParameterPtr<bool>>(),
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
}

template <typename T>
void defConstantParameter(py::module_& submodule, bool add_docs) {
  // Because this is an overloaded function, the docsstring will be rendered for
  // each overload. This option is to ensure that it can only be rendered for
  // the first one.
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
    return py::cast(UserSpecifiedParameter<data::UDTConfigPtr>::make(name));
  }

  throw std::invalid_argument("Invalid type '" +
                              py::str(type).cast<std::string>() +
                              "' passed to UserSpecifiedParameter. Must be one "
                              "of bool, int, float, str, or UDTConfig.");
}

}  // namespace thirdai::automl::deployment::python