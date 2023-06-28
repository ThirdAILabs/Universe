#include "Porting.h"
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/neuron_index/RandomSampler.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

py::object copyArray(const float* data, const std::vector<size_t>& shape) {
  NumpyArray<float> arr(shape);

  size_t total_dim =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<size_t>{});

  std::copy(data, data + total_dim, arr.mutable_data());

  return py::object(std::move(arr));
}

bool shapeMatches(const NumpyArray<float>& arr,
                  const std::vector<size_t>& expected_shape) {
  size_t ndim = arr.ndim();
  if (ndim != expected_shape.size()) {
    return false;
  }

  for (size_t i = 0; i < ndim; i++) {
    size_t dim = arr.shape(i);
    if (dim != expected_shape[i]) {
      return false;
    }
  }

  return true;
}

py::dict fullyConnectedOpParams(const ops::FullyConnectedPtr& fc) {
  py::dict params;

  params["type"] = "fully_connected";

  params["dim"] = fc->dim();
  params["input_dim"] = fc->inputDim();
  params["sparsity"] = fc->getSparsity();
  params["activation"] =
      activationFunctionToStr(fc->kernel()->getActivationFunction());
  params["use_bias"] = fc->kernel()->useBias();

  params["weights"] = copyArray(fc->weightsPtr(), {fc->dim(), fc->inputDim()});
  params["biases"] = copyArray(fc->biasesPtr(), {fc->dim()});

  params["rebuild_hash_tables"] = fc->getRebuildHashTables();
  params["reconstruct_hash_functions"] = fc->getReconstructHashFunctions();

  auto [hash_fn, hash_table] = fc->getHashTable();

  if (hash_fn && hash_table) {
    params["hash_fn"] = hash_fn;
    params["hash_table"] = hash_table;
    params["hash_table_frozen"] = fc->kernel()->isNeuronIndexFrozen();
  } else {
    if (std::dynamic_pointer_cast<RandomSampler>(fc->kernel()->neuronIndex())) {
      params["random_sampling"] = true;
    }
  }

  return params;
}

ops::FullyConnectedPtr fullyConnectedOpFromParams(const py::dict& params) {
  size_t dim = params["dim"].cast<size_t>();
  size_t input_dim = params["input_dim"].cast<size_t>();
  float sparsity = params["sparsity"].cast<float>();
  std::string activation = params["activation"].cast<std::string>();
  bool use_bias = params["use_bias"].cast<bool>();

  size_t rebuild_hash_tables = params["rebuild_hash_tables"].cast<size_t>();
  size_t reconstruct_hash_functions =
      params["reconstruct_hash_functions"].cast<size_t>();

  auto weights = params["weights"].cast<NumpyArray<float>>();
  auto biases = params["biases"].cast<NumpyArray<float>>();

  SamplingConfigPtr sampling = nullptr;

  if (params.contains("random_sampling") &&
      params["random_sampling"].cast<bool>()) {
    sampling = std::make_shared<RandomSamplingConfig>();
  }

  auto fc = ops::FullyConnected::make(dim, input_dim, sparsity, activation,
                                      sampling, use_bias, rebuild_hash_tables,
                                      reconstruct_hash_functions);

  if (!shapeMatches(weights, {dim, input_dim})) {
    throw std::invalid_argument("Invalid shape for weights. Expected (" +
                                std::to_string(dim) + ", " +
                                std::to_string(input_dim) + ").");
  }

  if (!shapeMatches(biases, {dim})) {
    throw std::invalid_argument("Invalid shape for biases. Expected (" +
                                std::to_string(dim) + ",).");
  }

  fc->setWeights(weights.data());
  fc->setBiases(biases.data());

  if (params.contains("hash_fn") && params.contains("hash_table")) {
    auto hash_fn = params["hash_fn"].cast<hashing::HashFunctionPtr>();
    auto hash_table =
        params["hash_table"].cast<hashtable::SampledHashTablePtr>();

    fc->setHashTable(hash_fn, hash_table);

    if (params["hash_table_frozen"].cast<bool>()) {
      fc->freezeHashTables(true);
    }
  }

  return fc;
}

py::dict embeddingOpParams(const ops::EmbeddingPtr& emb) {
  py::dict params;

  params["type"] = "embedding";

  params["dim"] = emb->dim();
  params["input_dim"] = emb->inputDim();
  params["activation"] = activationFunctionToStr(emb->activation());
  params["use_bias"] = emb->useBias();

  params["embeddings"] =
      copyArray(emb->embeddingsPtr(), {emb->inputDim(), emb->dim()});
  params["biases"] = copyArray(emb->biasesPtr(), {emb->dim()});

  return params;
}

ops::EmbeddingPtr embeddingOpFromParams(const py::dict& params) {
  size_t dim = params["dim"].cast<size_t>();
  size_t input_dim = params["input_dim"].cast<size_t>();
  std::string activation = params["activation"].cast<std::string>();
  bool use_bias = params["use_bias"].cast<bool>();

  auto embeddings = params["embeddings"].cast<NumpyArray<float>>();
  auto biases = params["biases"].cast<NumpyArray<float>>();

  auto emb = ops::Embedding::make(dim, input_dim, activation, use_bias);

  if (!shapeMatches(embeddings, {input_dim, dim})) {
    throw std::invalid_argument("Invalid shape for embeddings. Expected (" +
                                std::to_string(input_dim) + ", " +
                                std::to_string(dim) + ").");
  }

  if (!shapeMatches(biases, {dim})) {
    throw std::invalid_argument("Invalid shape for biases. Expected (" +
                                std::to_string(dim) + ",).");
  }

  emb->setEmbeddings(embeddings.data());
  emb->setBiases(biases.data());

  return emb;
}

py::dict modelParams(const model::ModelPtr& model) {
  if (model->inputDims().size() != 1 || model->outputs().size() != 1 ||
      model->losses().size() != 1 || model->labelDims().size() > 2) {
    throw std::runtime_error(
        "Converting model to parameters diction is currently only supported "
        "for models with a single input, output, and loss.");
  }

  py::dict params;

  params["input_dim"] = model->inputDims().front();

  if (model->labelDims().size() == 2) {
    params["mach_extra_label_dim"] = model->labelDims().back();
  }

  py::list ops;
  for (const auto& comp : model->computationOrder()) {
    if (auto fc = ops::FullyConnected::cast(comp->op())) {
      ops.append(fullyConnectedOpParams(fc));
    } else if (auto emb = ops::Embedding::cast(comp->op())) {
      ops.append(embeddingOpParams(emb));
    } else if (!std::dynamic_pointer_cast<ops::Input>(comp->op())) {
      throw std::invalid_argument(
          "Converting model to parameters dictionary is currently only "
          "supported for FullyConnected and Embedding Ops.");
    }
  }

  params["ops"] = ops;

  auto loss = model->losses().front();

  if (std::dynamic_pointer_cast<loss::CategoricalCrossEntropy>(loss)) {
    params["loss"] = "categorical_cross_entropy";
  } else if (std::dynamic_pointer_cast<loss::BinaryCrossEntropy>(loss)) {
    params["loss"] = "binary_cross_entropy";
  }

  params["train_steps"] = model->trainSteps();

  return params;
}

model::ModelPtr modelFromParams(const py::dict& params) {
  auto input = ops::Input::make(params["input_dim"].cast<uint32_t>());

  autograd::ComputationPtr output = input;
  for (auto op_params : params["ops"]) {
    auto op_params_dict = op_params.cast<py::dict>();

    std::string type = op_params_dict["type"].cast<std::string>();

    if (type == "fully_connected") {
      auto op = fullyConnectedOpFromParams(op_params_dict);
      output = op->apply(output);
    } else if (type == "embedding") {
      auto op = embeddingOpFromParams(op_params_dict);
      output = op->apply(output);
    } else {
      throw std::invalid_argument("Unexpected op type '" + type + "'.");
    }
  }

  auto labels = ops::Input::make(output->dim());

  std::string loss_name = params["loss"].cast<std::string>();

  loss::LossPtr loss;
  if (loss_name == "categorical_cross_entropy") {
    loss = loss::CategoricalCrossEntropy::make(output, labels);
  } else if (loss_name == "binary_cross_entropy") {
    loss = loss::BinaryCrossEntropy::make(output, labels);
  } else {
    throw std::invalid_argument("Unexpected loss '" + loss_name + "'.");
  }

  autograd::ComputationList additional_labels;
  if (params.contains("mach_extra_label_dim")) {
    additional_labels.push_back(
        ops::Input::make(params["mach_extra_label_dim"].cast<uint32_t>()));
  }

  auto model = model::Model::make({input}, {output}, {loss}, additional_labels);

  model->overrideTrainSteps(params["train_steps"].cast<size_t>());

  return model;
}

}  // namespace thirdai::bolt::nn::python