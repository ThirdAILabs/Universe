#include "ModelConfig.h"
#include "Parameter.h"
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::config {

/**
 * Helper function to create the sampling config for a fully connected layer.
 * Expects the config to either be a json object containing the keys
 * 'num_tables', 'hashes_per_table', and 'reservoir_size' or to simply be the
 * string 'random' indicating random sampling should be used.
 */
bolt::SamplingConfigPtr getSamplingConfig(const json& config,
                                          const ArgumentMap& args) {
  if (config.contains("sampling_config")) {
    const auto& sampling_json = config["sampling_config"];

    if (sampling_json.is_string()) {
      if (sampling_json.get<std::string>() == "random") {
        return std::make_shared<bolt::RandomSamplingConfig>();
      }

      if (sampling_json.get<std::string>() == "experimental_autotune") {
        uint32_t dim = integerParameter(config, "dim", args);
        float sparsity = floatParameter(config, "sparsity", args);
        return bolt::DWTASamplingConfig::newAutotune(dim, sparsity);
      }
    }

    if (sampling_json.is_object()) {
      uint32_t num_tables = integerParameter(sampling_json, "num_tables", args);
      uint32_t hashes_per_table =
          integerParameter(sampling_json, "hashes_per_table", args);
      uint32_t range_pow = integerParameter(sampling_json, "range_pow", args);
      uint32_t binsize = integerParameter(sampling_json, "binsize", args);
      uint32_t reservoir_size =
          integerParameter(sampling_json, "reservoir_size", args);
      uint32_t permutations =
          integerParameter(sampling_json, "permutations", args);

      return std::make_shared<bolt::DWTASamplingConfig>(
          num_tables, hashes_per_table, range_pow, binsize, reservoir_size,
          permutations);
    }
    throw std::invalid_argument(
        "Parameter 'sampling_config' must be a string 'random' indicating "
        "random sampling is used, or a string 'experimental_autotune' "
        "indicating experimental DWTA autotuner is used, or an object "
        "providing sampling parameters.");
  }

  return nullptr;
}

using CreatedComputations =
    std::unordered_map<std::string, bolt::ComputationPtr>;

/**
 * Helper function to get the predecessor of a node.
 */
bolt::ComputationPtr getPredecessor(const json& config,
                                    const CreatedComputations& created_comps) {
  std::string predecessor = getString(config, "predecessor");
  if (!created_comps.count(predecessor)) {
    throw std::invalid_argument("Could not find node '" + predecessor + "'.");
  }
  return created_comps.at(predecessor);
}

/**
 * Helper function to create a fully connected node. Expects the fields
 * 'dim', 'sparsity', 'activation', and 'predecessor' to be present in the
 * config. Optionally a field 'sampling_config' can be specified. If it is
 * not present the sampling parameters will be autotuned if not specified.
 */
bolt::ComputationPtr buildFullyConnected(
    const json& config, const ArgumentMap& args,
    const CreatedComputations& created_comps) {
  uint32_t dim = integerParameter(config, "dim", args);
  float sparsity = floatParameter(config, "sparsity", args);
  std::string activation = stringParameter(config, "activation", args);

  auto sampling_config = getSamplingConfig(config, args);

  auto predecessor = getPredecessor(config, created_comps);

  bool use_bias = true;
  if (config.contains("use_bias")) {
    use_bias = booleanParameter(config, "use_bias", args);
  }

  auto layer = bolt::FullyConnected::make(
      dim, predecessor->dim(), sparsity, activation, sampling_config, use_bias);

  return layer->applyUnary(predecessor);
}

/**
 * Helper function to create an embedding node. Expects the fields
 * 'num_embedding_lookups', 'lookup_size', 'log_embedding_block_size', and
 * 'reduction' to be present in the config. Optionally a field
 * 'num_tokens_per_input' can be specified to indicate the number of tokens in
 * each sample. This field is only required for concatenation reductions.
 */
bolt::ComputationPtr buildRobeZ(const json& config, const ArgumentMap& args,
                                const CreatedComputations& created_comps) {
  uint32_t num_lookups =
      integerParameter(config, "num_embedding_lookups", args);
  uint32_t lookup_size = integerParameter(config, "lookup_size", args);
  uint32_t log_block_size =
      integerParameter(config, "log_embedding_block_size", args);
  std::string reduction = stringParameter(config, "reduction", args);

  std::optional<uint32_t> num_tokens_per_input = std::nullopt;
  if (config.contains("num_tokens_per_input")) {
    num_tokens_per_input =
        integerParameter(config, "num_tokens_per_input", args);
  }

  auto layer = bolt::RobeZ::make(num_lookups, lookup_size, log_block_size,
                                 reduction, num_tokens_per_input);

  return layer->applyUnary(getPredecessor(config, created_comps));
}

bolt::ComputationPtr buildEmbedding(const json& config, const ArgumentMap& args,
                                    const CreatedComputations& created_comps) {
  uint32_t dim = integerParameter(config, "dim", args);

  std::string activation = stringParameter(config, "activation", args);

  auto predecessor = getPredecessor(config, created_comps);

  bool use_bias = true;
  if (config.contains("use_bias")) {
    use_bias = booleanParameter(config, "use_bias", args);
  }

  auto layer =
      bolt::Embedding::make(dim, predecessor->dim(), activation, use_bias);

  return layer->applyUnary(predecessor);
}

bolt::ComputationPtr buildLayerNorm(const json& config,
                                    const ArgumentMap& /*args*/,
                                    const CreatedComputations& created_comps) {
  auto predecessor = getPredecessor(config, created_comps);

  auto layer = bolt::LayerNorm::make();

  return layer->applyUnary(predecessor);
}

bolt::ComputationPtr buildActivation(const json& config,
                                     const ArgumentMap& args,
                                     const CreatedComputations& created_comps) {
  (void)args;

  auto predecessor = getPredecessor(config, created_comps);

  std::string type = text::lower(getString(config, "activation"));

  if (type == "relu") {
    return bolt::Relu::make()->applyUnary(predecessor);
  }
  if (type == "tanh") {
    return bolt::Tanh::make()->applyUnary(predecessor);
  }

  throw std::invalid_argument("Invalid activation type '" + type +
                              "'. Please use 'relu' or 'tanh'.");
}

/**
 * Helper function to construct the inputs. Matches the input dims to the
 * input names provided in the config. Updates created nodes to contain the
 * created inputs.
 */
bolt::ComputationList getInputs(const json& config,
                                const std::vector<uint32_t>& input_dims,
                                CreatedComputations& created_comps) {
  auto json_inputs = getArray(config, "inputs");
  if (config["inputs"].size() != input_dims.size()) {
    throw std::invalid_argument(
        "Expected inputs to be an array of input names of equal size to the "
        "number of input dims provided to the model.");
  }

  bolt::ComputationList inputs;
  for (uint32_t i = 0; i < input_dims.size(); i++) {
    inputs.push_back(bolt::Input::make(input_dims[i]));

    if (!json_inputs[i].is_string()) {
      throw std::invalid_argument("Expect inputs to be an array of strings.");
    }
    created_comps[json_inputs[i].get<std::string>()] = inputs.back();
  }
  return inputs;
}

bolt::ModelPtr buildModel(const json& config, const ArgumentMap& args,
                          const std::vector<uint32_t>& input_dims, bool mach) {
  CreatedComputations created_comps;

  auto inputs = getInputs(config, input_dims, created_comps);

  for (const auto& node_config : getArray(config, "nodes")) {
    if (!node_config.is_object()) {
      throw std::invalid_argument("Node config must be an json object.");
    }

    std::string name = getString(node_config, "name");

    std::string type = getString(node_config, "type");
    if (type == "fully_connected") {
      created_comps[name] =
          buildFullyConnected(node_config, args, created_comps);
    } else if (type == "robez") {
      created_comps[name] = buildRobeZ(node_config, args, created_comps);
    } else if (type == "embedding") {
      created_comps[name] = buildEmbedding(node_config, args, created_comps);
    } else if (type == "layernorm") {
      created_comps[name] = buildLayerNorm(node_config, args, created_comps);
    } else if (type == "activation") {
      created_comps[name] = buildActivation(node_config, args, created_comps);
    } else {
      throw std::invalid_argument("Found unsupported node type '" + type +
                                  "'.");
    }
  }

  std::string output_name = getString(config, "output");
  if (!created_comps.count(output_name)) {
    throw std::invalid_argument("Could not find node '" + output_name + "'.");
  }
  auto output = created_comps.at(output_name);

  auto labels = bolt::Input::make(output->dim());

  bolt::LossPtr loss;
  std::string loss_name = getString(config, "loss");

  if (text::lower(loss_name) == "categoricalcrossentropyloss") {
    loss = bolt::CategoricalCrossEntropy::make(output, labels);
  } else if (text::lower(loss_name) == "binarycrossentropyloss") {
    loss = bolt::BinaryCrossEntropy::make(output, labels);
  } else {
    throw std::invalid_argument("Invalid loss function '" + loss_name +
                                "' provided in model config.");
  }

  bolt::ComputationList additional_labels;
  if (mach) {
    // For mach we need the hash based labels for training, but the actual
    // document/class ids to compute metrics. Hence we add two labels to the
    // model.
    additional_labels.push_back(
        bolt::Input::make(std::numeric_limits<uint32_t>::max()));
  }

  auto model = bolt::Model::make(inputs, {output}, {loss}, additional_labels);

  return model;
}

/**
 * This is a helper function to both encrypt and decrypt a config by XORing each
 * byte in the config string with a byte from a random cipher. This ensures that
 * the config is not human readable, and also that the same byte will be
 * unlikely to have the same encoded value in different places (unless they a
 * seperated by a number of bytes that is a multiple of the cipher). Because of
 * the nature of XOR this function can both encrypt and decrypt the configs.
 */
std::string xorConfig(const std::string& config) {
  std::vector<uint8_t> cipher = {
      198, 31,  146, 160, 113, 8,   62,  25,  83,  68,  67,  97,  227, 93,  243,
      19,  159, 186, 148, 247, 179, 62,  2,   203, 211, 147, 105, 192, 220, 195,
      118, 145, 34,  168, 110, 75,  87,  28,  121, 221, 213, 102, 70,  141, 189,
      67,  126, 157, 139, 38,  210, 56,  245, 159, 146, 210, 189, 196, 121, 45,
      144, 233, 81,  82,  218, 55,  57,  114, 24,  180, 30,  127, 36,  43,  38,
      9,   157, 244, 224, 102, 159, 83,  213, 42,  172, 150, 187, 24,  77,  185,
      90,  8,   221, 77,  147, 130, 137, 172, 14,  111};

  std::string output;

  for (uint32_t i = 0; i < config.size(); i++) {
    output.push_back(config[i] ^ cipher[i % cipher.size()]);
  }

  return output;
}

void dumpConfig(const std::string& config, const std::string& filename) {
  std::ofstream file = dataset::SafeFileIO::ofstream(filename);
  file << xorConfig(config);
}

std::string loadConfig(const std::string& filename) {
  std::ifstream file = dataset::SafeFileIO::ifstream(filename);

  std::stringstream buffer;
  buffer << file.rdbuf();

  return xorConfig(buffer.str());
}

}  // namespace thirdai::automl::config