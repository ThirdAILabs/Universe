#include "Porting.h"
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/neuron_index/RandomSampler.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/SampledHashTable.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace thirdai::bolt::python {

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

using OpApplyFunc = std::function<ComputationPtr(
    const OpPtr& op, const ComputationList& inputs)>;

class OpConverter {
 public:
  virtual std::optional<py::dict> toParams(const OpPtr& op) const = 0;

  virtual std::pair<OpPtr, OpApplyFunc> fromParams(
      const py::dict& op) const = 0;

  virtual std::string opType() const = 0;

  virtual ~OpConverter() = default;

  template <typename OP_TYPE>
  static OpApplyFunc getUnaryApplyFunc() {
    auto apply_func = [](const OpPtr& op, const ComputationList& inputs) {
      auto concrete_op = std::dynamic_pointer_cast<OP_TYPE>(op);
      if (!concrete_op) {
        throw std::runtime_error("Op type mismatch in apply func.");
      }

      return concrete_op->applyUnary(inputs.at(0));
    };

    return apply_func;
  }

  py::dict emptyParams() const {
    py::dict params;
    params["type"] = opType();
    return params;
  }
};

class FullyConnectedOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    auto fc = FullyConnected::cast(op);
    if (!fc) {
      return std::nullopt;
    }

    py::dict params;

    params["type"] = opType();

    params["dim"] = fc->dim();
    params["input_dim"] = fc->inputDim();
    params["sparsity"] = fc->getSparsity();
    params["activation"] =
        activationFunctionToStr(fc->kernel()->getActivationFunction());
    params["use_bias"] = fc->kernel()->useBias();

    params["weights"] =
        copyArray(fc->weightsPtr(), {fc->dim(), fc->inputDim()});
    params["biases"] = copyArray(fc->biasesPtr(), {fc->dim()});

    params["rebuild_hash_tables"] = fc->getRebuildHashTables();
    params["reconstruct_hash_functions"] = fc->getReconstructHashFunctions();

    auto [hash_fn, hash_table] = fc->getHashTable();

    if (hash_fn && hash_table) {
      params["hash_fn"] = hash_fn;
      params["hash_table"] = hash_table;
      params["hash_table_frozen"] = fc->kernel()->isNeuronIndexFrozen();
    } else {
      if (std::dynamic_pointer_cast<RandomSampler>(
              fc->kernel()->neuronIndex())) {
        params["random_sampling"] = true;
      }
    }

    return params;
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
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

    auto fc = FullyConnected::make(dim, input_dim, sparsity, activation,
                                   sampling, use_bias, rebuild_hash_tables,
                                   reconstruct_hash_functions);

    if (!shapeMatches(weights, {dim, input_dim})) {
      throw std::invalid_argument(
          "Invalid shape for weights in FullyConnected op. Expected (" +
          std::to_string(dim) + ", " + std::to_string(input_dim) + ").");
    }

    if (!shapeMatches(biases, {dim})) {
      throw std::invalid_argument(
          "Invalid shape for biases in FullyConnected op. Expected (" +
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

    return {fc, getUnaryApplyFunc<FullyConnected>()};
  }

  std::string opType() const final { return "fully_connected"; }
};

class EmbeddingOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    auto emb = Embedding::cast(op);
    if (!emb) {
      return std::nullopt;
    }

    py::dict params;

    params["type"] = opType();

    params["dim"] = emb->dim();
    params["input_dim"] = emb->inputDim();
    params["activation"] = activationFunctionToStr(emb->activation());
    params["use_bias"] = emb->useBias();

    params["embeddings"] =
        copyArray(emb->embeddingsPtr(), {emb->inputDim(), emb->dim()});
    params["biases"] = copyArray(emb->biasesPtr(), {emb->dim()});

    return params;
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
    size_t dim = params["dim"].cast<size_t>();
    size_t input_dim = params["input_dim"].cast<size_t>();
    std::string activation = params["activation"].cast<std::string>();
    bool use_bias = params["use_bias"].cast<bool>();

    auto embeddings = params["embeddings"].cast<NumpyArray<float>>();
    auto biases = params["biases"].cast<NumpyArray<float>>();

    auto emb = Embedding::make(dim, input_dim, activation, use_bias);

    if (!shapeMatches(embeddings, {input_dim, dim})) {
      throw std::invalid_argument(
          "Invalid shape for embeddings in Embedding op. Expected (" +
          std::to_string(input_dim) + ", " + std::to_string(dim) + ").");
    }

    if (!shapeMatches(biases, {dim})) {
      throw std::invalid_argument(
          "Invalid shape for biases in Embedding op. Expected (" +
          std::to_string(dim) + ",).");
    }

    emb->setEmbeddings(embeddings.data());
    emb->setBiases(biases.data());

    return {emb, getUnaryApplyFunc<Embedding>()};
  }

  std::string opType() const final { return "embedding"; }
};

class RobeZOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    auto emb = std::dynamic_pointer_cast<RobeZ>(op);
    if (!emb) {
      return std::nullopt;
    }

    py::dict params;

    params["type"] = opType();

    params["num_embedding_lookups"] = emb->kernel()->numEmbeddingLookups();
    params["lookup_size"] = emb->kernel()->lookupSize();
    params["log_embedding_block_size"] = emb->kernel()->logEmbeddingBlockSize();
    params["reduction"] = emb->kernel()->reduction();
    params["num_tokens_per_input"] = emb->kernel()->numTokensPerInput();
    params["update_chunk_size"] = emb->kernel()->updateChunkSize();
    params["hash_seed"] = emb->kernel()->hashSeed();

    size_t emb_block_size = emb->kernel()->getRawEmbeddingBlock().size();
    const float* emb_block_ptr = emb->kernel()->getRawEmbeddingBlock().data();

    auto arr =
        copyArray(emb_block_ptr, {emb_block_size}).cast<NumpyArray<float>>();
    params["embedding_block"] = arr;

    return params;
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
    size_t num_embedding_lookups =
        params["num_embedding_lookups"].cast<size_t>();
    size_t lookup_size = params["lookup_size"].cast<size_t>();
    size_t log_embedding_block_size =
        params["log_embedding_block_size"].cast<size_t>();
    std::string reduction = params["reduction"].cast<std::string>();
    std::optional<size_t> num_tokens_per_input =
        params["num_tokens_per_input"].cast<std::optional<size_t>>();
    size_t update_chunk_size = params["update_chunk_size"].cast<size_t>();
    uint32_t hash_seed = params["hash_seed"].cast<uint32_t>();

    auto embedding_block = params["embedding_block"].cast<NumpyArray<float>>();

    auto emb = RobeZ::make(num_embedding_lookups, lookup_size,
                           log_embedding_block_size, reduction,
                           num_tokens_per_input, update_chunk_size, hash_seed);

    size_t embedding_block_size = emb->kernel()->getRawEmbeddingBlock().size();
    if (!shapeMatches(embedding_block, {embedding_block_size})) {
      throw std::invalid_argument(
          "Expected embedding block to be 1D array of size " +
          std::to_string(embedding_block_size) + ".");
    }

    std::copy(embedding_block.data(),
              embedding_block.data() + embedding_block_size,
              emb->kernel()->getRawEmbeddingBlock().data());

    return {emb, getUnaryApplyFunc<RobeZ>()};
  }

  std::string opType() const final { return "robe_z"; }
};

class LayerNormOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    auto norm = std::dynamic_pointer_cast<LayerNorm>(op);
    if (!norm) {
      return std::nullopt;
    }

    py::dict params;

    params["type"] = opType();

    params["gamma"] = copyArray(norm->gamma().data(), {norm->dim()});
    params["beta"] = copyArray(norm->beta().data(), {norm->dim()});

    return params;
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
    auto gamma = params["gamma"].cast<NumpyArray<float>>();
    auto beta = params["beta"].cast<NumpyArray<float>>();

    if (gamma.ndim() != 1 || beta.ndim() != 1 ||
        gamma.shape(0) != beta.shape(0)) {
      throw std::invalid_argument(
          "Expected gamma and beta to be 1D arrays of the same size.");
    }

    auto op = LayerNorm::make(gamma.data(), beta.data(), gamma.shape(0));

    return {op, getUnaryApplyFunc<LayerNorm>()};
  }

  std::string opType() const final { return "layer_norm"; }
};

class ConcatenateOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    if (!std::dynamic_pointer_cast<Concatenate>(op)) {
      return std::nullopt;
    }

    return emptyParams();
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
    (void)params;

    auto op = Concatenate::make();

    auto apply_func = [](const OpPtr& op, const ComputationList& inputs) {
      auto concrete_op = std::dynamic_pointer_cast<Concatenate>(op);
      if (!concrete_op) {
        throw std::runtime_error("Op type mismatch in apply func.");
      }

      return concrete_op->apply(inputs);
    };

    return {op, apply_func};
  }

  std::string opType() const final { return "concat"; }
};

class TanhOpConverter final : public OpConverter {
 public:
  std::optional<py::dict> toParams(const OpPtr& op) const final {
    if (!std::dynamic_pointer_cast<Tanh>(op)) {
      return std::nullopt;
    }

    return emptyParams();
  }

  std::pair<OpPtr, OpApplyFunc> fromParams(const py::dict& params) const final {
    (void)params;

    auto op = Tanh::make();

    return {op, getUnaryApplyFunc<Tanh>()};
  }

  std::string opType() const final { return "tanh"; }
};

std::vector<std::shared_ptr<OpConverter>> op_converters = {
    std::make_shared<FullyConnectedOpConverter>(),
    std::make_shared<EmbeddingOpConverter>(),
    std::make_shared<RobeZOpConverter>(),
    std::make_shared<LayerNormOpConverter>(),
    std::make_shared<ConcatenateOpConverter>(),
    std::make_shared<TanhOpConverter>(),
};

py::list exportPlaceholders(const ComputationList& placeholders) {
  py::list placeholder_infos;
  for (const auto& placeholder : placeholders) {
    py::dict placeholder_info;
    placeholder_info["name"] = placeholder->name();
    placeholder_info["dim"] = placeholder->dim();
    placeholder_infos.append(placeholder_info);
  }

  return placeholder_infos;
}

py::dict exportOps(const ModelPtr& model) {
  py::dict ops;
  for (const auto& op : model->ops()) {
    bool found_converter = false;
    for (const auto& converter : op_converters) {
      if (auto params = converter->toParams(op)) {
        ops[py::str(op->name())] = std::move(params.value());
        found_converter = true;
        break;
      }
    }

    if (!found_converter) {
      throw std::invalid_argument("Unable to find op converter for op '" +
                                  op->name() + "'.");
    }
  }
  return ops;
}

py::list exportComputations(const ModelPtr& model) {
  py::list computations;
  for (const auto& comp : model->computationOrderWithoutInputs()) {
    py::dict comp_info;
    comp_info["name"] = comp->name();
    comp_info["op"] = comp->op()->name();

    py::list comp_inputs;
    for (const auto& input : comp->inputs()) {
      comp_inputs.append(input->name());
    }
    comp_info["inputs"] = comp_inputs;

    computations.append(comp_info);
  }

  return computations;
}

py::list exportOutputs(const ModelPtr& model) {
  py::list outputs;

  for (const auto& output : model->outputs()) {
    outputs.append(output->name());
  }

  return outputs;
}

py::list exportLosses(const ModelPtr& model) {
  py::list losses;
  for (const auto& loss : model->losses()) {
    py::dict loss_info;

    if (std::dynamic_pointer_cast<CategoricalCrossEntropy>(loss)) {
      loss_info["name"] = "categorical_cross_entropy";
      loss_info["output"] = loss->outputsUsed().at(0)->name();
      loss_info["label"] = loss->labels().at(0)->name();
    } else if (std::dynamic_pointer_cast<BinaryCrossEntropy>(loss)) {
      loss_info["name"] = "binary_cross_entropy";
      loss_info["output"] = loss->outputsUsed().at(0)->name();
      loss_info["label"] = loss->labels().at(0)->name();
    }
    losses.append(loss_info);
  }
  return losses;
}

py::dict modelParams(const ModelPtr& model) {
  py::dict model_params;
  model_params["ops"] = exportOps(model);
  model_params["inputs"] = exportPlaceholders(model->inputs());
  model_params["labels"] = exportPlaceholders(model->labels());
  model_params["computation_graph"] = exportComputations(model);
  model_params["losses"] = exportLosses(model);
  model_params["outputs"] = exportOutputs(model);
  model_params["train_steps"] = model->trainSteps();

  return model_params;
}

ComputationList importPlaceholders(
    const py::list& placeholder_infos,
    std::unordered_map<std::string, ComputationPtr>& computations) {
  ComputationList placeholders;
  for (const auto& placeholder_info : placeholder_infos) {
    std::string name = placeholder_info["name"].cast<std::string>();
    if (!computations.count(name)) {
      size_t dim = placeholder_info["dim"].cast<size_t>();

      auto placeholder = Input::make(dim);
      computations[name] = placeholder;
      placeholders.push_back(placeholder);
    }
  }

  return placeholders;
}

using OpMap = std::unordered_map<std::string, std::pair<OpPtr, OpApplyFunc>>;

using ComputationMap = std::unordered_map<std::string, ComputationPtr>;

OpMap importOps(const py::dict& params) {
  std::unordered_map<std::string, std::shared_ptr<OpConverter>> converter_map;
  for (const auto& converter : op_converters) {
    converter_map[converter->opType()] = converter;
  }

  OpMap ops;

  for (const auto& op_info : params["ops"].cast<py::dict>()) {
    std::string name = op_info.first.cast<std::string>();

    py::dict op_params = op_info.second.cast<py::dict>();

    std::string type = op_params["type"].cast<std::string>();

    if (!converter_map.count(type)) {
      throw std::invalid_argument("No converter found for op type '" + type +
                                  "'.");
    }
    auto [op, apply_func] = converter_map.at(type)->fromParams(op_params);
    op->setName(name);
    ops[name] = {op, apply_func};
  }

  return ops;
}

void importComputationGraph(const py::dict& params, const OpMap& ops,
                            ComputationMap& computations) {
  for (const auto& comp : params["computation_graph"]) {
    std::string name = comp["name"].cast<std::string>();
    auto [op, apply_func] = ops.at(comp["op"].cast<std::string>());

    ComputationList inputs;
    for (const auto input : comp["inputs"]) {
      inputs.push_back(computations[input.cast<std::string>()]);
    }

    computations[name] = apply_func(op, inputs);
  }
}

std::vector<LossPtr> importLosses(const py::dict& params,
                                  const ComputationMap& computations) {
  std::vector<LossPtr> losses;
  for (const auto& loss : params["losses"]) {
    std::string loss_name = loss["name"].cast<std::string>();

    if (loss_name == "categorical_cross_entropy") {
      auto output = loss["output"].cast<std::string>();
      auto label = loss["label"].cast<std::string>();
      auto loss = CategoricalCrossEntropy::make(computations.at(output),
                                                computations.at(label));
      losses.push_back(loss);
    } else if (loss_name == "binary_cross_entropy") {
      auto output = loss["output"].cast<std::string>();
      auto label = loss["label"].cast<std::string>();
      auto loss = BinaryCrossEntropy::make(computations.at(output),
                                           computations.at(label));
      losses.push_back(loss);
    } else {
      throw std::invalid_argument("Unexpected loss '" + loss_name + "'.");
    }
  }

  return losses;
}

ComputationList importOutputs(const py::dict& params,
                              const ComputationMap& computations) {
  ComputationList comps;
  for (const auto& output : params["outputs"]) {
    comps.push_back(computations.at(output.cast<std::string>()));
  }

  return comps;
}

ModelPtr modelFromParams(const py::dict& params) {
  OpMap ops = importOps(params);

  ComputationMap computations;
  auto inputs =
      importPlaceholders(params["inputs"].cast<py::list>(), computations);
  auto labels =
      importPlaceholders(params["labels"].cast<py::list>(), computations);
  importComputationGraph(params, ops, computations);
  auto losses = importLosses(params, computations);
  auto outputs = importOutputs(params, computations);

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

  auto model = Model::make(inputs, outputs, losses,
                           /* additional_labels= */ labels);

  model->overrideTrainSteps(params["train_steps"].cast<size_t>());

  return model;
}

}  // namespace thirdai::bolt::python