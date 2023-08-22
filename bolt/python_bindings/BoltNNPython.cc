#include "BoltNNPython.h"
#include "PybindUtils.h"
#include <bolt/python_bindings/Porting.h>
#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/loss/MarginBCE.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/CosineSimilarity.h>
#include <bolt/src/nn/ops/DlrmAttention.h>
#include <bolt/src/nn/ops/DotProduct.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/tensor/Tensor.h>
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

namespace thirdai::bolt::python {

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
py::object toNumpy(const TensorPtr& tensor, const T* data) {
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

TensorPtr fromNumpySparse(const NumpyArray<uint32_t>& indices,
                          const NumpyArray<float>& values, uint32_t last_dim) {
  if (indices.ndim() != 2) {
    throw std::invalid_argument("Expected indices to be 2D.");
  }
  if (values.ndim() != 2) {
    throw std::invalid_argument("Expected values to be 2D.");
  }

  uint32_t batch_size = indices.shape(0);
  uint32_t nonzeros = indices.shape(1);

  return Tensor::fromArray(indices.data(), values.data(), batch_size, last_dim,
                           nonzeros, /* with_grad= */ false);
}

TensorPtr fromNumpyDense(const NumpyArray<float>& values) {
  if (values.ndim() != 2) {
    throw std::invalid_argument("Expected values to be 2D.");
  }

  uint32_t batch_size = values.shape(0);
  uint32_t dim = values.shape(1);

  return Tensor::fromArray(nullptr, values.data(), batch_size, dim,
                           /* nonzeros= */ dim, /* with_grad= */ false);
}

void defineTensor(py::module_& nn);

void defineOps(py::module_& nn);

void defineLosses(py::module_& nn);

void createBoltNNSubmodule(py::module_& module) {
  auto nn = module.def_submodule("nn");

  py::class_<Model, ModelPtr>(nn, "Model")
#if THIRDAI_EXPOSE_ALL
      /**
       * ==============================================================
       * WARNING: If this THIRDAI_EXPOSE_ALL is removed then license
       * checks must be added to train_on_batch. Also methods such as
       * summary, ops, __getitem__, etc. should remain hidden.
       * ==============================================================
       */
      .def(py::init(&Model::make), py::arg("inputs"), py::arg("outputs"),
           py::arg("losses"), py::arg("additional_labels") = ComputationList{})
      .def("train_on_batch", &Model::trainOnBatch, py::arg("inputs"),
           py::arg("labels"))
      .def("forward",
           py::overload_cast<const TensorList&, bool>(&Model::forward),
           py::arg("inputs"), py::arg("use_sparsity") = false)
      .def("update_parameters", &Model::updateParameters,
           py::arg("learning_rate"))
      .def("ops", &Model::opExecutionOrder)
      .def("__getitem__", &Model::getOp, py::arg("name"))
      .def("outputs", &Model::outputs)
      .def("labels", &Model::labels)
      .def("summary", &Model::summary, py::arg("print") = true)
      .def("num_params", &Model::numParams)
      .def("get_parameters", &getParameters,
           py::return_value_policy::reference_internal)
      .def("set_parameters", &setParameters, py::arg("new_values"))
      .def("train_steps", &Model::trainSteps)
      .def("override_train_steps", &Model::overrideTrainSteps,
           py::arg("train_steps"))
      .def("params", &modelParams)
      .def_static("from_params", &modelFromParams, py::arg("params"))
#endif
      // The next three functions are used for distributed training.
      .def("disable_sparse_parameter_updates",
           &Model::disableSparseParameterUpdates)
      .def("get_gradients", &getGradients,
           py::return_value_policy::reference_internal)
      .def("set_gradients", &setGradients, py::arg("new_values"))
      .def("enable_sparse_parameter_updates",
           &Model::enableSparseParameterUpdates)
      .def("freeze_hash_tables", &Model::freezeHashTables,
           py::arg("insert_labels_if_not_found") = true)
      .def("unfreeze_hash_tables", &Model::unfreezeHashTables)
      .def("save", &Model::save, py::arg("filename"),
           py::arg("save_metadata") = true)
      .def("checkpoint", &Model::checkpoint, py::arg("filename"),
           py::arg("save_metadata") = true)
      .def_static("load", &Model::load, py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<Model>());

#if THIRDAI_EXPOSE_ALL
  defineTensor(nn);

  defineOps(nn);

  defineLosses(nn);
#endif
}

void defineTensor(py::module_& nn) {
  py::class_<Tensor, TensorPtr>(nn, "Tensor")
      .def(py::init([](BoltVector vector, uint32_t dim) {
             return Tensor::convert(std::move(vector), dim);
           }),
           py::arg("vector"), py::arg("dim"))
      .def(py::init(&fromNumpySparse), py::arg("indices"), py::arg("values"),
           py::arg("dense_dim"))
      .def(py::init(&fromNumpyDense), py::arg("values"))
      .def("__getitem__", &Tensor::getVector)
      .def("__len__", &Tensor::batchSize)
      .def_property_readonly(
          "active_neurons",
          [](const TensorPtr& tensor) {
            return toNumpy(tensor, tensor->activeNeuronsPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "activations",
          [](const TensorPtr& tensor) {
            return toNumpy(tensor, tensor->activationsPtr());
          },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "gradients",
          [](const TensorPtr& tensor) {
            return toNumpy(tensor, tensor->gradientsPtr());
          },
          py::return_value_policy::reference_internal);
}

void defineOps(py::module_& nn) {
#if THIRDAI_EXPOSE_ALL
#pragma message("THIRDAI_EXPOSE_ALL is defined")  // NOLINT

  py::class_<Computation, ComputationPtr>(nn, "Computation")
      .def("dim", &Computation::dim)
      .def("tensor", &Computation::tensor)
      .def("name", &Computation::name);

  py::class_<Op, OpPtr>(nn, "Op")
      .def("dim", &Op::dim)
      .def_property("name", &Op::name, &Op::setName);

  py::class_<thirdai::bolt::SamplingConfig, SamplingConfigPtr>(  // NOLINT
      nn, "SamplingConfig");

  py::class_<thirdai::bolt::DWTASamplingConfig,
             std::shared_ptr<DWTASamplingConfig>, SamplingConfig>(
      nn, "DWTASamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                    uint32_t>(),
           py::arg("num_tables"), py::arg("hashes_per_table"),
           py::arg("range_pow"), py::arg("binsize"), py::arg("reservoir_size"),
           py::arg("permutations"));

  py::class_<thirdai::bolt::FastSRPSamplingConfig,
             std::shared_ptr<FastSRPSamplingConfig>, SamplingConfig>(
      nn, "FastSRPSamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("reservoir_size"));

  py::class_<RandomSamplingConfig, std::shared_ptr<RandomSamplingConfig>,
             SamplingConfig>(nn, "RandomSamplingConfig")
      .def(py::init<>());

  py::class_<hashtable::SampledHashTable, hashtable::SampledHashTablePtr>(
      nn, "HashTable")
      .def("save", &hashtable::SampledHashTable::save, py::arg("filename"))
      .def_static("load", &hashtable::SampledHashTable::load,
                  py::arg("filename"))
      .def(thirdai::bolt::python::getPickleFunction<
           hashtable::SampledHashTable>());
#endif

  py::class_<FullyConnected, FullyConnectedPtr, Op>(nn, "FullyConnected")
      .def(py::init(&FullyConnected::make), py::arg("dim"),
           py::arg("input_dim"), py::arg("sparsity") = 1.0,
           py::arg("activation") = "relu", py::arg("sampling_config") = nullptr,
           py::arg("use_bias") = true, py::arg("rebuild_hash_tables") = 10,
           py::arg("reconstruct_hash_functions") = 100)
      .def("__call__", &FullyConnected::apply)
      .def("dim", &FullyConnected::dim)
      .def("get_sparsity", &FullyConnected::getSparsity)
      .def("set_sparsity", &FullyConnected::setSparsity, py::arg("sparsity"),
           py::arg("rebuild_hash_tables") = true,
           py::arg("experimental_autotune") = false)
      .def_property_readonly(
          "weights",
          [](const FullyConnected& op) {
            return toNumpy(op.weightsPtr(), {op.dim(), op.inputDim()});
          })
      .def_property_readonly("biases",
                             [](const FullyConnected& op) {
                               return toNumpy(op.biasesPtr(), {op.dim()});
                             })
      .def("set_weights",
           [](FullyConnected& op, const NumpyArray<float>& weights) {
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
           [](FullyConnected& op, const NumpyArray<float>& biases) {
             if (biases.ndim() != 1 || biases.shape(0) != op.dim()) {
               std::stringstream error;
               error << "Expected biases to be 1D array with shape ("
                     << op.dim() << ",).";
               throw std::invalid_argument(error.str());
             }
             op.setBiases(biases.data());
           })
      .def("get_hash_table", &FullyConnected::getHashTable)
      .def("set_hash_table", &FullyConnected::setHashTable, py::arg("hash_fn"),
           py::arg("hash_table"));

  py::class_<RobeZ, RobeZPtr, Op>(nn, "RobeZ")
      .def(py::init(&RobeZ::make), py::arg("num_embedding_lookups"),
           py::arg("lookup_size"), py::arg("log_embedding_block_size"),
           py::arg("reduction"), py::arg("num_tokens_per_input") = std::nullopt,
           py::arg("update_chunk_size") = DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE,
           py::arg("seed") = global_random::nextSeed())
      .def("__call__", &RobeZ::apply)
      .def("duplicate_with_new_reduction", &RobeZ::duplicateWithNewReduction,
           py::arg("reduction"), py::arg("num_tokens_per_input"));

  py::class_<Embedding, EmbeddingPtr, Op>(nn, "Embedding")
      .def(py::init(&Embedding::make), py::arg("dim"), py::arg("input_dim"),
           py::arg("activation"), py::arg("bias") = true)
      .def("__call__", &Embedding::apply)
      .def_property_readonly(
          "weights",
          [](const EmbeddingPtr& op) {
            return toNumpy(op->embeddingsPtr(), {op->inputDim(), op->dim()});
          })
      .def_property_readonly("biases",
                             [](const EmbeddingPtr& op) {
                               return toNumpy(op->biasesPtr(), {op->dim()});
                             })
      .def("set_weights",
           [](EmbeddingPtr& op, const NumpyArray<float>& weights) {
             if (weights.ndim() != 2 || weights.shape(0) != op->inputDim() ||
                 weights.shape(1) != op->dim()) {
               std::stringstream error;
               error << "Expected weights to be 2D array with shape ("
                     << op->inputDim() << ", " << op->dim() << ").";
               throw std::invalid_argument(error.str());
             }
             op->setEmbeddings(weights.data());
           })
      .def("set_biases", [](EmbeddingPtr& op, const NumpyArray<float>& biases) {
        if (biases.ndim() != 1 || biases.shape(0) != op->dim()) {
          std::stringstream error;
          error << "Expected biases to be 1D array with shape (" << op->dim()
                << ",).";
          throw std::invalid_argument(error.str());
        }
        op->setBiases(biases.data());
      });

  py::class_<Concatenate, ConcatenatePtr, Op>(nn, "Concatenate")
      .def(py::init(&Concatenate::make))
      .def("__call__", &Concatenate::apply);

  py::class_<LayerNorm, LayerNormPtr, Op>(nn, "LayerNorm")
      .def(py::init(py::overload_cast<>(&LayerNorm::make)))
      .def("__call__", &LayerNorm::apply);

  py::class_<Tanh, TanhPtr, Op>(nn, "Tanh")
      .def(py::init(&Tanh::make))
      .def("__call__", &Tanh::apply);

  py::class_<Relu, ReluPtr, Op>(nn, "Relu")
      .def(py::init(&Relu::make))
      .def("__call__", &Relu::apply);

  py::class_<DotProduct, DotProductPtr, Op>(nn, "DotProduct")
      .def(py::init<>(&DotProduct::make))
      .def("__call__", &DotProduct::apply);

  py::class_<CosineSimilarity, CosineSimilarityPtr, Op>(nn, "CosineSimilarity")
      .def(py::init<>(&CosineSimilarity::make),
           py::arg("clipping_mode") = std::nullopt)
      .def("__call__", &CosineSimilarity::apply);

  py::class_<DlrmAttention, DlrmAttentionPtr, Op>(nn, "DlrmAttention")
      .def(py::init(&DlrmAttention::make))
      .def("__call__", &DlrmAttention::apply);

  nn.def("Input", &Input::make, py::arg("dim"));
}

void defineLosses(py::module_& nn) {
  auto loss = nn.def_submodule("losses");

  py::class_<Loss, LossPtr>(loss, "Loss");  // NOLINT

  py::class_<CategoricalCrossEntropy, CategoricalCrossEntropyPtr, Loss>(
      loss, "CategoricalCrossEntropy")
      .def(py::init(&CategoricalCrossEntropy::make), py::arg("activations"),
           py::arg("labels"));

  py::class_<MarginBCE, MarginBCEPtr, Loss>(loss, "MarginBCE")
      .def(py::init(&MarginBCE::make), py::arg("activations"),
           py::arg("labels"));

  py::class_<BinaryCrossEntropy, BinaryCrossEntropyPtr, Loss>(
      loss, "BinaryCrossEntropy")
      .def(py::init(&BinaryCrossEntropy::make), py::arg("activations"),
           py::arg("labels"));

  py::class_<EuclideanContrastive, EuclideanContrastivePtr, Loss>(
      loss, "EuclideanContrastive")
      .def(py::init(&EuclideanContrastive::make), py::arg("output_1"),
           py::arg("output_2"), py::arg("labels"),
           py::arg("dissimilar_cutoff_distance"));
}

}  // namespace thirdai::bolt::python