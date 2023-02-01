#include "ModelConfig.h"
#include "Parameter.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <fstream>
#include <random>
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

    if (sampling_json.is_string() &&
        sampling_json.get<std::string>() == "random") {
      return std::make_shared<bolt::RandomSamplingConfig>();
    }
    if (sampling_json.is_object()) {
      uint32_t num_tables = integerParameter(sampling_json, "num_tables", args);
      uint32_t hashes_per_table =
          integerParameter(sampling_json, "hashes_per_table", args);
      uint32_t reservoir_size =
          integerParameter(sampling_json, "reservoir_size", args);

      return std::make_shared<bolt::DWTASamplingConfig>(
          num_tables, hashes_per_table, reservoir_size);
    }
    throw std::invalid_argument(
        "Parameter 'sampling_config' must be a string 'random' indicating "
        "random sampling is used or an object providing sampling parameters.");
  }

  return nullptr;
}

/**
 * Helper function to create a fully connected node. Expects the fields 'dim',
 * 'sparsity', 'activation', and 'predecessor' to be present in the config.
 * Optionally a field 'sampling_config' can be specified. If it is not present
 * the sampling parameters will be autotuned if not specified.
 */
bolt::NodePtr buildFullyConnectedNode(
    const json& config, const ArgumentMap& args,
    const std::unordered_map<std::string, bolt::NodePtr>& created_nodes) {
  uint32_t dim = integerParameter(config, "dim", args);
  float sparsity = floatParameter(config, "sparsity", args);
  std::string activation = stringParameter(config, "activation", args);

  auto sampling_config = getSamplingConfig(config, args);

  bolt::FullyConnectedNodePtr layer;
  if (sampling_config) {
    layer = bolt::FullyConnectedNode::make(dim, sparsity, activation,
                                           sampling_config);
  } else {
    layer = bolt::FullyConnectedNode::makeAutotuned(dim, sparsity, activation);
  }

  std::string predecessor = getString(config, "predecessor");
  if (!created_nodes.count(predecessor)) {
    throw std::invalid_argument("Could not find node '" + predecessor + "'.");
  }
  layer->addPredecessor(created_nodes.at(predecessor));

  return layer;
}

/**
 * Helper function to construct the inputs. Matches the input dims to the input
 * names provided in the config. Updates created nodes to contain the created
 * inputs.
 */
std::vector<bolt::InputPtr> getInputs(
    const json& config, const std::vector<uint32_t>& input_dims,
    std::unordered_map<std::string, bolt::NodePtr>& created_nodes) {
  auto json_inputs = getArray(config, "inputs");
  if (config["inputs"].size() != input_dims.size()) {
    throw std::invalid_argument(
        "Expected inputs to be an array of input names of equal size to the "
        "number of input dims provided to the model.");
  }

  std::vector<bolt::InputPtr> inputs;
  for (uint32_t i = 0; i < input_dims.size(); i++) {
    inputs.push_back(bolt::Input::make(input_dims[i]));

    if (!json_inputs[i].is_string()) {
      throw std::invalid_argument("Expect inputs to be an array of strings.");
    }
    created_nodes[json_inputs[i].get<std::string>()] = inputs.back();
  }
  return inputs;
}

bolt::BoltGraphPtr buildModel(const json& config, const ArgumentMap& args,
                              const std::vector<uint32_t>& input_dims) {
  std::unordered_map<std::string, bolt::NodePtr> created_nodes;

  auto inputs = getInputs(config, input_dims, created_nodes);

  for (const auto& node_config : getArray(config, "nodes")) {
    if (!node_config.is_object()) {
      throw std::invalid_argument("Node config must be an json object.");
    }

    std::string name = getString(node_config, "name");

    std::string type = getString(node_config, "type");
    if (type == "fully_connected") {
      created_nodes[name] =
          buildFullyConnectedNode(node_config, args, created_nodes);
    } else {
      throw std::invalid_argument("Found unsupported node type '" + type +
                                  "'.");
    }
  }

  std::string output = getString(config, "output");
  if (!created_nodes.count(output)) {
    throw std::invalid_argument("Could not find node '" + output + "'.");
  }

  auto loss = bolt::getLossFunction(getString(config, "loss"));

  auto model =
      std::make_shared<bolt::BoltGraph>(inputs, created_nodes.at(output));

  model->compile(loss);

  return model;
}

/**
 * This is a helper function to both encrypt and decrypt a config by XORing each
 * byte in the config string with a random byte from a random generator. This
 * ensures that the config is not human readable, and also that the same byte
 * will be unlikely to have the same encoded value in different places. Because
 * of the nature of XOR this function can both encrypt and decrypt the configs.
 * Because the seed is fixed the sequence of random bytes for XORing will always
 * be the same, this is essentially a shortcut for having to store a long byte
 * string in the code for encrypting/decrypting configs.
 */
std::string xorConfig(const std::string& config) {
  std::mt19937 rand(8256387);
  std::uniform_int_distribution<uint8_t> dist;

  std::string output;

  for (char c : config) {
    output.push_back(c ^ dist(rand));
  }

  return output;
}

void dumpConfig(const std::string& config, const std::string& filename) {
  std::ofstream file(filename);
  file << xorConfig(config);
}

std::string loadConfig(const std::string& filename) {
  std::ifstream file(filename);

  std::stringstream buffer;
  buffer << file.rdbuf();

  return xorConfig(buffer.str());
}

}  // namespace thirdai::automl::config