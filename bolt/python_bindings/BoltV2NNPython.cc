#include "BoltV2NNPython.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

template <typename T>
py::object toNumpy(const T* data, std::vector<uint32_t> shape) {
  if (data) {
    py::array_t<T, py::array::c_style | py::array::forcecast> arr(shape, data);
    return py::object(std::move(arr));
  }
  return py::none();
}

void createBoltV2NNSubmodule(py::module_& module) {
  auto nn = module.def_submodule("nn");

  py::class_<tensor::Tensor, tensor::TensorPtr>(nn, "Tensor");  // NOLINT

  py::class_<tensor::InputTensor, tensor::InputTensorPtr, tensor::Tensor>(
      nn, "Input")
      .def(py::init(&tensor::InputTensor::make), py::arg("dim"),
           py::arg("sparse_nonzeros") = std::nullopt);

  py::class_<tensor::ActivationTensor, tensor::ActivationTensorPtr,
             tensor::Tensor>(nn, "ActivationTensor")
      .def_property_readonly(
          "active_neurons",
          [](const tensor::ActivationTensor& tensor) {
            return toNumpy(tensor.activeNeuronsPtr(), tensor.shape());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "activations",
          [](const tensor::ActivationTensor& tensor) {
            return toNumpy(tensor.activationsPtr(), tensor.shape());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "gradients",
          [](const tensor::ActivationTensor& tensor) {
            return toNumpy(tensor.gradientsPtr(), tensor.shape());
          },
          py::return_value_policy::reference_internal);

  py::class_<ops::Op, ops::OpPtr>(nn, "Op");  // NOLINT

  py::class_<ops::FullyConnected, ops::FullyConnectedPtr, ops::Op>(
      nn, "FullyConnected")
      .def(py::init(&ops::FullyConnected::make), py::arg("dim"),
           py::arg("input_dim"), py::arg("sparsity") = 1.0,
           py::arg("activation") = "relu", py::arg("sampling_config") = nullptr,
           py::arg("rebuild_hash_tables") = 10,
           py::arg("reconstruct_hash_functions") = 100)
      .def("__call__", &ops::FullyConnected::apply)
      .def_property_readonly("weights",
                             [](const ops::FullyConnected& op) {
                               return toNumpy(op.weightsPtr(), op.dimensions());
                             })
      .def_property_readonly("biases", [](const ops::FullyConnected& op) {
        return toNumpy(op.biasesPtr(), {op.dimensions()[0]});
      });

  py::class_<model::Model, model::ModelPtr>(nn, "Model")
      .def(py::init(&model::Model::make), py::arg("inputs"), py::arg("outputs"),
           py::arg("losses"))
      .def("train_on_batch", &model::Model::trainOnBatchSingleInput,
           py::arg("inputs"), py::arg("labels"))
      .def("train_on_batch", &model::Model::trainOnBatch, py::arg("inputs"),
           py::arg("labels"))
      .def("forward", &model::Model::forwardSingleInput, py::arg("inputs"),
           py::arg("use_sparsity"))
      .def("forward", &model::Model::forward, py::arg("inputs"),
           py::arg("use_sparsity"))
      .def("update_parameters", &model::Model::updateParameters,
           py::arg("learning_rate"))
      .def("__getitem__", &model::Model::getOp, py::arg("name"))
      .def("summary", &model::Model::summary, py::arg("print") = true);

  auto loss = nn.def_submodule("losses");

  py::class_<loss::Loss, loss::LossPtr>(loss, "Loss");  // NOLINT

  py::class_<loss::CategoricalCrossEntropy, loss::CategoricalCrossEntropyPtr,
             loss::Loss>(loss, "CategoricalCrossEntropy")
      .def(py::init(&loss::CategoricalCrossEntropy::make),
           py::arg("activations"));

  py::class_<loss::BinaryCrossEntropy, loss::BinaryCrossEntropyPtr, loss::Loss>(
      loss, "BinaryCrossEntropy")
      .def(py::init(&loss::BinaryCrossEntropy::make), py::arg("activations"));
}

}  // namespace thirdai::bolt::nn::python