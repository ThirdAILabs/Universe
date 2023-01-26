#include "ModelConfig.h"
#include "Parameter.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/config/ParameterInputMap.h>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::config {

bolt::SamplingConfigPtr getSamplingConfig(const json& config,
                                          const ParameterInputMap& user_input) {
  if (config.contains("sampling_config")) {
    const auto& sampling_json = config["sampling_config"];

    if (sampling_json.is_string() &&
        sampling_json.get<std::string>() == "random") {
      return std::make_shared<bolt::RandomSamplingConfig>();
    }
    if (sampling_json.is_object()) {
      uint32_t num_tables =
          integerParameter(sampling_json, "num_tables", user_input);
      uint32_t hashes_per_table =
          integerParameter(sampling_json, "hashes_per_table", user_input);
      uint32_t reservoir_size =
          integerParameter(sampling_json, "reservoir_size", user_input);

      return std::make_shared<bolt::DWTASamplingConfig>(
          num_tables, hashes_per_table, reservoir_size);
    }
    throw std::invalid_argument(
        "Parameter 'sampling_config' must be a string 'random' indicating "
        "random sampling is used or an object providing sampling parameters.");
  }

  return nullptr;
}

bolt::NodePtr buildFullyConnectedNode(
    const json& config, const ParameterInputMap& user_input,
    const std::unordered_map<std::string, bolt::NodePtr>& created_nodes) {
  uint32_t dim = integerParameter(config, "dim", user_input);
  float sparsity = floatParameter(config, "sparsity", user_input);
  std::string activation = stringParameter(config, "activation", user_input);

  auto sampling_config = getSamplingConfig(config, user_input);

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

bolt::BoltGraphPtr buildModel(const json& config,
                              const ParameterInputMap& user_input,
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
          buildFullyConnectedNode(node_config, user_input, created_nodes);
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