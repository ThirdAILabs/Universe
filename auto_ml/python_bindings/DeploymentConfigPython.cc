#include "DeploymentConfigPython.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <iostream>
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

  submodule.def("test", [](const py::object& obj) {
    std::cout << "Testing: " << py::str(obj) << std::endl;
    if (py::isinstance<py::str>(obj)) {
      std::cout << "is str" << std::endl;
    }
    if (py::isinstance<py::int_>(obj)) {
      std::cout << "is int" << std::endl;
    }
    if (py::isinstance<py::float_>(obj)) {
      std::cout << "is float" << std::endl;
    }
    if (py::isinstance<bolt::SamplingConfig>(obj)) {
      std::cout << "is sampling config" << std::endl;
    }
    if (py::isinstance<bolt::LossFunction>(obj)) {
      std::cout << "is loss function" << std::endl;
    }
  });
}

template<typename P, typename T>
py::object castToHyperParameterPyObject(std::unique_ptr<P<T>> )

py::object createConstantParameter(const py::object& obj) {
  if (py::isinstance<py::str>(obj)) {
    std::string str = obj.cast<std::string>();
    if (str.length() == 1) {
      return std::make_unique<ConstantParameter<char>>(str[0]);
    }
  }
  if (py::isinstance<py::int_>(obj)) {
    std::cout << "is int" << std::endl;
  }
  if (py::isinstance<py::float_>(obj)) {
    std::cout << "is float" << std::endl;
  }
  if (py::isinstance<bolt::SamplingConfig>(obj)) {
    std::cout << "is sampling config" << std::endl;
  }
  if (py::isinstance<bolt::LossFunction>(obj)) {
    std::cout << "is loss function" << std::endl;
  }
}

}  // namespace thirdai::automl::deployment_config::python