#include "BoltV2NNPython.h"
#include "PybindUtils.h"
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/Tanh.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <licensing/src/methods/file/License.h>
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
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
py::object toNumpy(const T* data, std::vector<uint32_t> shape) {
  if (data) {
    NumpyArray<T> arr(shape, data);
    return py::object(std::move(arr));
  }
  return py::none();
}

template <typename T>
py::object toNumpy(const tensor::TensorPtr& tensor, const T* data) {
  auto dims = tensor->dims();
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }

  dims.back() = *nonzeros;

  if (data) {
    return toNumpy(data, dims);
  }

  // We return None if the data is nullptr so that a user can access the field
  // and check if its None rather than dealing with an exception. For example:
  // if tensor.active_neurons:
  //      do something
  return py::none();
}

tensor::TensorPtr fromNumpySparse(const NumpyArray<uint32_t>& indices,
                                  const NumpyArray<float>& values,
                                  uint32_t last_dim) {
  if (indices.ndim() != values.ndim()) {
    throw std::invalid_argument(
        "Expected indices and values to have the same number of dimensions.");
  }

  tensor::Dims dims;
  for (uint32_t i = 0; i < indices.ndim(); i++) {
    if (indices.shape(i) != values.shape(i)) {
      throw std::invalid_argument("Dimension mismatch in indices and values.");
    }
    dims.push_back(indices.shape(i));
  }
  uint32_t nonzeros = dims.back();
  dims.back() = last_dim;

  return tensor::Tensor::fromArray(indices.data(), values.data(), dims,
                                   nonzeros, /* with_grad= */ false);
}

tensor::TensorPtr fromNumpyDense(const NumpyArray<float>& values) {
  tensor::Dims dims;
  for (uint32_t i = 0; i < values.ndim(); i++) {
    dims.push_back(values.shape(i));
  }

  return tensor::Tensor::fromArray(nullptr, values.data(), dims,
                                   /* nonzeros= */ dims.back(),
                                   /* with_grad= */ false);
}

void defineTensor(py::module_& nn);

void defineOps(py::module_& nn);

void defineLosses(py::module_& nn);

void createBoltV2NNSubmodule(py::module_& module) {
  auto nn = module.def_submodule("nn");

  py::class_<model::Model, model::ModelPtr>(nn, "Model")
#if THIRDAI_EXPOSE_ALL
      /**
       * ==============================================================
       * WARNING: If this THIRDAI_EXPOSE_ALL is removed then license
       * checks must be added to train_on_batch. Also methods such as
       * summary, ops, __getitem__, etc. should remain hidden.
       * ==============================================================
       */
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
      .def("ops", &model::Model::opExecutionOrder)
      .def("__getitem__", &model::Model::getOp, py::arg("name"))
      .def("outputs", &model::Model::outputs)
      .def("labels", &model::Model::labels)
      .def("summary", &model::Model::summary, py::arg("print") = true)
#endif
      .def("save", &model::Model::save, py::arg("filename"),
           py::arg("save_metadata") = true)
      .def("checkpoint", &model::Model::checkpoint, py::arg("filename"),
           py::arg("save_metadata") = true)
      .def_static("load", &model::Model::load, py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<model::Model>());

#if THIRDAI_EXPOSE_ALL
  defineTensor(nn);

  defineOps(nn);

  defineLosses(nn);
#endif
}

void defineTensor(py::module_& nn) {
  py::class_<tensor::Tensor, tensor::TensorPtr>(nn, "Tensor")
      .def(py::init([](BoltVector vector, uint32_t dim) {
             return tensor::Tensor::convert(std::move(vector), dim);
           }),
           py::arg("vector"), py::arg("dim"))
      .def(py::init(&fromNumpySparse), py::arg("indices"), py::arg("values"),
           py::arg("dense_dim"))
      .def(py::init(&fromNumpyDense), py::arg("values"))
      .def("dims", &tensor::Tensor::dims)
      .def("sparse", &tensor::Tensor::isSparse)
      .def_property_readonly(
          "indices",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->indicesPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "values",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->valuesPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "gradients",
          [](const tensor::TensorPtr& tensor) {
            return toNumpy(tensor, tensor->gradientsPtr());
          },
          py::return_value_policy::reference_internal);
}

void defineOps(py::module_& nn) {
  py::class_<autograd::Computation, autograd::ComputationPtr>(nn, "Computation")
      .def("dims", &autograd::Computation::dims)
      .def("tensor", &autograd::Computation::tensor)
      .def("name", &autograd::Computation::name);

  py::class_<ops::Op, ops::OpPtr>(nn, "Op").def("name", &ops::Op::name);

  py::class_<ops::FullyConnected, ops::FullyConnectedPtr, ops::Op>(
      nn, "FullyConnected")
      .def(py::init(&ops::FullyConnected::make), py::arg("dim"),
           py::arg("input_dim"), py::arg("sparsity") = 1.0,
           py::arg("activation") = "relu", py::arg("sampling_config") = nullptr,
           py::arg("rebuild_hash_tables") = 10,
           py::arg("reconstruct_hash_functions") = 100)
      .def("__call__", &ops::FullyConnected::apply)
      .def("dim", &ops::FullyConnected::dim)
      .def_property_readonly(
          "weights",
          [](const ops::FullyConnected& op) {
            return toNumpy(op.weightsPtr(), {op.dim(), op.inputDim()});
          })
      .def_property_readonly("biases",
                             [](const ops::FullyConnected& op) {
                               return toNumpy(op.biasesPtr(), {op.dim()});
                             })
      .def("set_weights",
           [](ops::FullyConnected& op, const NumpyArray<float>& weights) {
             if (weights.ndim() != 2 || weights.shape(0) != op.dim() ||
                 weights.shape(1) != op.inputDim()) {
               std::stringstream error;
               error << "Expected weights to be 2D array with shape ("
                     << op.dim() << ", " << op.inputDim() << ").";
               throw std::invalid_argument(error.str());
             }
             op.setWeights(weights.data());
           })
      .def("set_biases",
           [](ops::FullyConnected& op, const NumpyArray<float>& biases) {
             if (biases.ndim() != 1 || biases.shape(0) != op.dim()) {
               std::stringstream error;
               error << "Expected biases to be 1D array with shape ("
                     << op.dim() << ",).";
               throw std::invalid_argument(error.str());
             }
             op.setBiases(biases.data());
           })
      .def("get_hash_table", &ops::FullyConnected::getHashTable)
      .def("set_hash_table", &ops::FullyConnected::setHashTable,
           py::arg("hash_fn"), py::arg("hash_table"));

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

  py::class_<ops::LayerNorm, ops::LayerNormPtr, ops::Op>(nn, "LayerNorm")
      .def(py::init(&ops::LayerNorm::make))
      .def("__call__", &ops::LayerNorm::apply);

  py::class_<ops::Tanh, ops::TanhPtr, ops::Op>(nn, "Tanh")
      .def(py::init(&ops::Tanh::make))
      .def("__call__", &ops::Tanh::apply);

  nn.def("Input", py::overload_cast<uint32_t>(&ops::Input::make),
         py::arg("dim"));

  nn.def("Input", py::overload_cast<std::vector<uint32_t>>(&ops::Input::make),
         py::arg("dims"));
}

void defineLosses(py::module_& nn) {
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

  py::class_<loss::EuclideanContrastive, loss::EuclideanContrastivePtr,
             loss::Loss>(loss, "EuclideanContrastive")
      .def(py::init(&loss::EuclideanContrastive::make), py::arg("output_1"),
           py::arg("output_2"), py::arg("labels"),
           py::arg("dissimilar_cutoff_distance"));
}

}  // namespace thirdai::bolt::nn::python