#include "BoltV2NNPython.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>

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

template <typename T>
py::object toNumpy(const tensor::TensorPtr& tensor, const T* data) {
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }
  if (data) {
    py::array_t<T, py::array::c_style | py::array::forcecast> arr(
        {tensor->batchSize(), *nonzeros}, data);
    return py::object(std::move(arr));
  }
  // We return None if the data is nullptr so that a user can access the field
  // and check if its None rather than dealing with an exception. For example:
  // if tensor.active_neurons:
  //      do something
  return py::none();
}

void createBoltV2NNSubmodule(py::module_& module) {
  auto nn = module.def_submodule("nn");

  py::class_<tensor::Tensor, tensor::TensorPtr>(nn, "Tensor")
      .def_property_readonly(
          "active_neurons",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->activeNeuronsPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "activations",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->activationsPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "gradients",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->gradientsPtr());
          },
          py::return_value_policy::reference_internal);

  py::class_<autograd::Computation, autograd::ComputationPtr>(nn, "Computation")
      .def("dim", &autograd::Computation::dim)
      .def("tensor", &autograd::Computation::tensor)
      .def("name", &autograd::Computation::name);

  py::class_<ops::Op, ops::OpPtr>(nn, "Op")
      .def("name", &ops::Op::name)
      .def("from_scratch", &ops::Op::fromScratch)
      .def("freeze", &ops::Op::freeze)
      .def("unfreeze", &ops::Op::unfreeze);

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

  py::class_<ops::Embedding, ops::EmbeddingPtr, ops::Op>(nn, "Embedding")
      .def(py::init(&ops::Embedding::make), py::arg("num_embedding_lookups"),
           py::arg("lookup_size"), py::arg("log_embedding_block_size"),
           py::arg("reduction"), py::arg("num_tokens_per_input") = std::nullopt,
           py::arg("update_chunk_size") = DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE)
      .def("__call__", &ops::Embedding::apply)
      .def("duplicate_with_new_reduction",
           &ops::Embedding::duplicateWithNewReduction, py::arg("reduction"),
           py::arg("num_tokens_per_input"));

  py::class_<ops::Concatenate, ops::ConcatenatePtr, ops::Op>(nn, "Concatenate")
      .def(py::init(&ops::Concatenate::make))
      .def("__call__", &ops::Concatenate::apply);

  nn.def("Input", &ops::Input::make, py::arg("dim"));

  py::class_<model::Model, model::ModelPtr>(nn, "Model")
      .def(py::init(&model::Model::make), py::arg("inputs"), py::arg("outputs"),
           py::arg("losses"))
      .def("train_on_batch", &model::Model::trainOnBatch, py::arg("inputs"),
           py::arg("labels"))
      .def("forward",
           py::overload_cast<const tensor::TensorList&, bool>(
               &model::Model::forward),
           py::arg("inputs"), py::arg("use_sparsity"))
      .def("update_parameters", &model::Model::updateParameters,
           py::arg("learning_rate"))
      .def("__getitem__", &model::Model::getOp, py::arg("name"))
      .def("summary", &model::Model::summary, py::arg("print") = true)
      .def("save", &model::Model::save, py::arg("filename"))
      .def_static("load", &model::Model::load, py::arg("filename"));

  auto loss = nn.def_submodule("losses");

  py::class_<loss::Loss, loss::LossPtr>(loss, "Loss");  // NOLINT

  py::class_<loss::CategoricalCrossEntropy, loss::CategoricalCrossEntropyPtr,
             loss::Loss>(loss, "CategoricalCrossEntropy")
      .def(py::init(&loss::CategoricalCrossEntropy::make),
           py::arg("activations"), py::arg("labels"));

  py::class_<loss::BinaryCrossEntropy, loss::BinaryCrossEntropyPtr, loss::Loss>(
      loss, "BinaryCrossEntropy")
      .def(py::init(&loss::BinaryCrossEntropy::make), py::arg("activations"),
           py::arg("labels"));
}

}  // namespace thirdai::bolt::nn::python