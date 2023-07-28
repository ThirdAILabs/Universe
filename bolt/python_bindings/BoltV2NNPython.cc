#include "BoltV2NNPython.h"
#include "PybindUtils.h"
#include <bolt/python_bindings/Porting.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/DlrmAttention.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/ops/Tanh.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <compression/python_bindings/ConversionUtils.h>
#include <compression/src/CompressionFactory.h>
#include <licensing/src/methods/file/License.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

template <typename T>
using NumpyArray = thirdai::bolt::python::NumpyArray<T>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

using FloatCompressedVector =
    std::variant<thirdai::compression::DragonVector<float>,
                 thirdai::compression::CountSketch<float>>;

template <typename T>
SerializedCompressedVector compress(const py::array_t<T>& data,
                                    const std::string& compression_scheme,
                                    float compression_density,
                                    uint32_t seed_for_hashing,
                                    uint32_t sample_population_size) {
  FloatCompressedVector compressed_vector = thirdai::compression::compress(
      data.data(), static_cast<uint32_t>(data.size()), compression_scheme,
      compression_density, seed_for_hashing, sample_population_size);

  uint32_t serialized_size =
      std::visit(thirdai::compression::SizeVisitor<float>(), compressed_vector);

  char* serialized_compressed_vector = new char[serialized_size];

  std::visit(thirdai::compression::SerializeVisitor<float>(
                 serialized_compressed_vector),
             compressed_vector);

  py::capsule free_when_done(serialized_compressed_vector,
                             [](void* ptr) { delete static_cast<char*>(ptr); });

  return SerializedCompressedVector(
      serialized_size, serialized_compressed_vector, free_when_done);
}

template <typename T>
py::array_t<T> decompress(SerializedCompressedVector& new_params) {
  const char* serialized_data =
      py::cast<SerializedCompressedVector>(new_params).data();
  FloatCompressedVector compressed_vector =
      thirdai::compression::python::deserializeCompressedVector<float>(
          serialized_data);
  std::vector<float> full_gradients = std::visit(
      thirdai::compression::DecompressVisitor<float>(), compressed_vector);

  return py::array_t<T>(full_gradients.size(), full_gradients.data());
}

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
  auto nonzeros = tensor->nonzeros();
  if (!nonzeros) {
    throw std::runtime_error(
        "Cannot convert tensor to numpy if the number of nonzeros is not "
        "fixed.");
  }
  if (data) {
    NumpyArray<T> arr({tensor->batchSize(), *nonzeros}, data);
    return py::object(std::move(arr));
  }
  // We return None if the data is nullptr so that a user can access the field
  // and check if its None rather than dealing with an exception. For example:
  // if tensor.active_neurons:
  //      do something
  return py::none();
}

void defineTensor(py::module_& nn);

void defineOps(py::module_& nn);

void defineLosses(py::module_& nn);

void createCompressionSubmodule(py::module_& module) {
  auto compression = module.def_submodule("compression");

  compression.def(
      "compress", &compress<float>, py::arg("data"),
      py::arg("compression_scheme"), py::arg("compression_density"),
      py::arg("seed_for_hashing"), py::arg("sample_population_size"),
      "Returns a char array representing a compressed vector. "
      "sample_population_size is the number of random samples you take "
      "for estimating a threshold for dragon compression or the number "
      "of sketches needed for count_sketch");

  compression.def("decompress", &decompress<float>, py::arg("new_params"));
}

void createBoltV2NNSubmodule(py::module_& module) {
  auto nn = module.def_submodule("nn");

  createCompressionSubmodule(nn);
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
           py::arg("losses"),
           py::arg("additional_labels") = autograd::ComputationList{})
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
      .def("get_parameters", &::thirdai::bolt::python::getParameters,
           py::return_value_policy::reference_internal)
      .def("set_parameters", &::thirdai::bolt::python::setParameters,
           py::arg("new_values"))
      .def("train_steps", &model::Model::trainSteps)
      .def("override_train_steps", &model::Model::overrideTrainSteps,
           py::arg("train_steps"))
      .def("params", &modelParams)
      .def_static("from_params", &modelFromParams, py::arg("params"))
#endif
      // The next three functions are used for distributed training.
      .def("disable_sparse_parameter_updates",
           &model::Model::disableSparseParameterUpdates)
      .def("get_gradients", &::thirdai::bolt::python::getGradients,
           py::return_value_policy::reference_internal)
      .def("set_gradients", &::thirdai::bolt::python::setGradients,
           py::arg("new_values"))
      .def("enable_sparse_parameter_updates",
           &model::Model::enableSparseParameterUpdates)
      .def("freeze_hash_tables", &model::Model::freezeHashTables,
           py::arg("insert_labels_if_not_found") = true)
      .def("unfreeze_hash_tables", &model::Model::unfreezeHashTables)
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
}

void defineOps(py::module_& nn) {
  py::class_<autograd::Computation, autograd::ComputationPtr>(nn, "Computation")
      .def("dim", &autograd::Computation::dim)
      .def("tensor", &autograd::Computation::tensor)
      .def("name", &autograd::Computation::name);

  py::class_<ops::Op, ops::OpPtr>(nn, "Op")
      .def("dim", &ops::Op::dim)
      .def_property("name", &ops::Op::name, &ops::Op::setName);

  py::class_<ops::FullyConnected, ops::FullyConnectedPtr, ops::Op>(
      nn, "FullyConnected")
      .def(py::init(&ops::FullyConnected::make), py::arg("dim"),
           py::arg("input_dim"), py::arg("sparsity") = 1.0,
           py::arg("activation") = "relu", py::arg("sampling_config") = nullptr,
           py::arg("use_bias") = true, py::arg("rebuild_hash_tables") = 10,
           py::arg("reconstruct_hash_functions") = 100)
      .def("__call__", &ops::FullyConnected::apply)
      .def("dim", &ops::FullyConnected::dim)
      .def("get_sparsity", &ops::FullyConnected::getSparsity)
      .def("set_sparsity", &ops::FullyConnected::setSparsity,
           py::arg("sparsity"), py::arg("rebuild_hash_tables") = true,
           py::arg("experimental_autotune") = false)
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

  py::class_<ops::RobeZ, ops::RobeZPtr, ops::Op>(nn, "RobeZ")
      .def(py::init(&ops::RobeZ::make), py::arg("num_embedding_lookups"),
           py::arg("lookup_size"), py::arg("log_embedding_block_size"),
           py::arg("reduction"), py::arg("num_tokens_per_input") = std::nullopt,
           py::arg("update_chunk_size") = DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE,
           py::arg("seed") = global_random::nextSeed())
      .def("__call__", &ops::RobeZ::apply)
      .def("duplicate_with_new_reduction",
           &ops::RobeZ::duplicateWithNewReduction, py::arg("reduction"),
           py::arg("num_tokens_per_input"));

  py::class_<ops::Embedding, ops::EmbeddingPtr, ops::Op>(nn, "Embedding")
      .def(py::init(&ops::Embedding::make), py::arg("dim"),
           py::arg("input_dim"), py::arg("activation"), py::arg("bias") = true)
      .def("__call__", &ops::Embedding::apply)
      .def_property_readonly(
          "weights",
          [](const ops::EmbeddingPtr& op) {
            return toNumpy(op->embeddingsPtr(), {op->inputDim(), op->dim()});
          })
      .def_property_readonly("biases",
                             [](const ops::EmbeddingPtr& op) {
                               return toNumpy(op->biasesPtr(), {op->dim()});
                             })
      .def("set_weights",
           [](ops::EmbeddingPtr& op, const NumpyArray<float>& weights) {
             if (weights.ndim() != 2 || weights.shape(0) != op->inputDim() ||
                 weights.shape(1) != op->dim()) {
               std::stringstream error;
               error << "Expected weights to be 2D array with shape ("
                     << op->inputDim() << ", " << op->dim() << ").";
               throw std::invalid_argument(error.str());
             }
             op->setEmbeddings(weights.data());
           })
      .def("set_biases",
           [](ops::EmbeddingPtr& op, const NumpyArray<float>& biases) {
             if (biases.ndim() != 1 || biases.shape(0) != op->dim()) {
               std::stringstream error;
               error << "Expected biases to be 1D array with shape ("
                     << op->dim() << ",).";
               throw std::invalid_argument(error.str());
             }
             op->setBiases(biases.data());
           });

  py::class_<ops::Concatenate, ops::ConcatenatePtr, ops::Op>(nn, "Concatenate")
      .def(py::init(&ops::Concatenate::make))
      .def("__call__", &ops::Concatenate::apply);

  py::class_<ops::LayerNorm, ops::LayerNormPtr, ops::Op>(nn, "LayerNorm")
      .def(py::init(py::overload_cast<>(&ops::LayerNorm::make)))
      .def("__call__", &ops::LayerNorm::apply);

  py::class_<ops::Tanh, ops::TanhPtr, ops::Op>(nn, "Tanh")
      .def(py::init(&ops::Tanh::make))
      .def("__call__", &ops::Tanh::apply);

  py::class_<ops::DlrmAttention, ops::DlrmAttentionPtr, ops::Op>(
      nn, "DlrmAttention")
      .def(py::init(&ops::DlrmAttention::make))
      .def("__call__", &ops::DlrmAttention::apply);

  nn.def("Input", &ops::Input::make, py::arg("dim"));
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