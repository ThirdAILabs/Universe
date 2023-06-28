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
#include <string>
#include <unordered_map>

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

class ModelExporter {
 public:
  static py::dict run(model::ModelPtr model) {
    ModelExporter exp(std::move(model));
    return exp.exportModel();
  }

 private:
  explicit ModelExporter(model::ModelPtr model) : _model(std::move(model)) {}

  py::dict exportModel() {
    py::dict model_params;
    model_params["ops"] = exportOps();
    model_params["inputs"] = exportPlacholders(_model->inputs());
    model_params["labels"] = exportPlacholders(_model->labels());
    model_params["computation_graph"] = exportComputations();
    model_params["losses"] = exportLosses();
    model_params["outputs"] = exportOutputs();
    model_params["train_steps"] = _model->trainSteps();

    return model_params;
  }

  py::list exportPlacholders(const autograd::ComputationList& placeholders) {
    py::list placeholder_infos;
    for (const auto& placeholder : placeholders) {
      std::string name = nextCompName();
      _comps_to_names[placeholder] = name;

      py::dict placeholder_info;
      placeholder_info["name"] = name;
      placeholder_info["dim"] = placeholder->dim();
      placeholder_infos.append(placeholder_info);
    }

    return placeholder_infos;
  }

  py::dict exportOps() {
    py::dict ops;
    for (const auto& op : _model->ops()) {
      if (auto fc = ops::FullyConnected::cast(op)) {
        ops[py::str(op->name())] = fullyConnectedOpParams(fc);
      } else if (auto emb = ops::Embedding::cast(op)) {
        ops[py::str(op->name())] = embeddingOpParams(emb);
      } else {
        throw std::invalid_argument(
            "Converting model to parameters dictionary is currently only "
            "supported for FullyConnected and Embedding Ops.");
      }
    }
    return ops;
  }

  py::list exportComputations() {
    py::list computations;
    for (const auto& comp : _model->computationOrderWithoutInputs()) {
      std::string name = nextCompName();
      _comps_to_names[comp] = name;

      py::dict comp_info;
      comp_info["name"] = name;
      comp_info["op"] = comp->op()->name();

      py::list comp_inputs;
      for (const auto& input : comp->inputs()) {
        comp_inputs.append(getName(input));
      }
      comp_info["inputs"] = comp_inputs;

      computations.append(comp_info);
    }

    return computations;
  }

  py::list exportLosses() {
    py::list losses;
    for (const auto& loss : _model->losses()) {
      py::dict loss_info;

      if (std::dynamic_pointer_cast<loss::CategoricalCrossEntropy>(loss)) {
        loss_info["name"] = "categorical_cross_entropy";
        loss_info["output"] = getName(loss->outputsUsed().at(0));
        loss_info["label"] = getName(loss->labels().at(0));
      } else if (std::dynamic_pointer_cast<loss::BinaryCrossEntropy>(loss)) {
        loss_info["name"] = "binary_cross_entropy";
        loss_info["output"] = getName(loss->outputsUsed().at(0));
        loss_info["label"] = getName(loss->labels().at(0));
      }
      losses.append(loss_info);
    }
    return losses;
  }

  py::list exportOutputs() {
    py::list outputs;

    for (const auto& output : _model->outputs()) {
      outputs.append(getName(output));
    }

    return outputs;
  }

  std::string nextCompName() const {
    return "computation_" + std::to_string(_comps_to_names.size());
  }

  std::string getName(const autograd::ComputationPtr& comp) {
    if (!_comps_to_names.count(comp)) {
      throw std::runtime_error("No matching name for computation: '" +
                               comp->name() + "'.");
    }
    return _comps_to_names.at(comp);
  }

  model::ModelPtr _model;

  std::unordered_map<autograd::ComputationPtr, std::string> _comps_to_names;
};

class ModelImporter {
 public:
  static model::ModelPtr run(py::dict params) {
    ModelImporter imp(std::move(params));
    return imp.importModel();
  }

 private:
  explicit ModelImporter(py::dict params) : _params(std::move(params)) {}

  model::ModelPtr importModel() {
    importOps();

    auto inputs = importPlaceholders(_params["inputs"].cast<py::list>());
    auto labels = importPlaceholders(_params["labels"].cast<py::list>());
    importComputationGraph();

    auto losses = importLosses();
    auto outputs = importOutputs();

    // Remove all labels used by the losses, the rest are additional labels such
    // as the ones used for metrics in Mach.
    for (const auto& loss : losses) {
      for (const auto& label : loss->labels()) {
        auto loc = std::find(labels.begin(), labels.end(), label);
        if (loc != labels.end()) {
          labels.erase(loc);
        }
      }
    }

    auto model = model::Model::make(inputs, outputs, losses,
                                    /* additional_labels= */ labels);

    model->overrideTrainSteps(_params["train_steps"].cast<size_t>());

    return model;
  }

  autograd::ComputationList importPlaceholders(
      const py::list& placeholder_infos) {
    autograd::ComputationList placeholders;
    for (const auto& placeholder_info : placeholder_infos) {
      std::string name = placeholder_info["name"].cast<std::string>();
      size_t dim = placeholder_info["dim"].cast<size_t>();
      auto placeholder = ops::Input::make(dim);
      _computations[name] = placeholder;
      placeholders.push_back(placeholder);
    }

    return placeholders;
  }

  void importOps() {
    for (const auto& op_info : _params["ops"].cast<py::dict>()) {
      std::string name = op_info.first.cast<std::string>();

      py::dict op_params = op_info.second.cast<py::dict>();

      std::string type = op_params["type"].cast<std::string>();
      if (type == "fully_connected") {
        _ops[name] = fullyConnectedOpFromParams(op_params);
      } else if (type == "embedding") {
        _ops[name] = embeddingOpFromParams(op_params);
      } else {
        throw std::invalid_argument("Unexpected op type '" + type + "'.");
      }
    }
  }

  void importComputationGraph() {
    for (const auto& comp : _params["computation_graph"]) {
      std::string name = comp["name"].cast<std::string>();
      ops::OpPtr op = _ops[comp["op"].cast<std::string>()];

      autograd::ComputationList inputs;
      for (const auto input : comp["inputs"]) {
        inputs.push_back(_computations[input.cast<std::string>()]);
      }

      if (auto fc = ops::FullyConnected::cast(op)) {
        _computations[name] = fc->apply(inputs.at(0));
      } else if (auto emb = ops::Embedding::cast(op)) {
        _computations[name] = emb->apply(inputs.at(0));
      } else {
        throw std::runtime_error("Unsupported op in model parameters.");
      }
    }
  }

  std::vector<loss::LossPtr> importLosses() {
    std::vector<loss::LossPtr> losses;
    for (const auto& loss : _params["losses"]) {
      std::string loss_name = loss["name"].cast<std::string>();

      if (loss_name == "categorical_cross_entropy") {
        auto output = loss["output"].cast<std::string>();
        auto label = loss["label"].cast<std::string>();
        auto loss = loss::CategoricalCrossEntropy::make(getComp(output),
                                                        getComp(label));
        losses.push_back(loss);
      } else if (loss_name == "binary_cross_entropy") {
        auto output = loss["output"].cast<std::string>();
        auto label = loss["label"].cast<std::string>();
        auto loss =
            loss::BinaryCrossEntropy::make(getComp(output), getComp(label));
        losses.push_back(loss);
      } else {
        throw std::invalid_argument("Unexpected loss '" + loss_name + "'.");
      }
    }

    return losses;
  }

  autograd::ComputationList importOutputs() {
    autograd::ComputationList comps;
    for (const auto& output : _params["outputs"]) {
      comps.push_back(_computations[output.cast<std::string>()]);
    }

    return comps;
  }

  autograd::ComputationPtr getComp(const std::string& name) {
    if (!_computations.count(name)) {
      throw std::runtime_error("No matching computation for name: '" + name +
                               "'.");
    }
    return _computations.at(name);
  }

  py::dict _params;

  std::unordered_map<std::string, ops::OpPtr> _ops;
  std::unordered_map<std::string, autograd::ComputationPtr> _computations;
};

py::dict modelParams(const model::ModelPtr& model) {
  return ModelExporter::run(model);
}

model::ModelPtr modelFromParams(const py::dict& params) {
  return ModelImporter::run(params);
}

}  // namespace thirdai::bolt::nn::python