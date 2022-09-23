#include "DeploymentConfigPython.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>
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

}  // namespace thirdai::automl::deployment_config::python