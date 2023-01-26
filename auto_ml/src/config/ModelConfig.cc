#include "ModelConfig.h"
#include "Parameter.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::config {

bolt::NodePtr buildFullyConnectedNode(
    const json& config, const ParameterInputMap& user_input,
    const std::unordered_map<std::string, bolt::NodePtr>& created_nodes) {
  uint32_t dim = integerParameter(config, "dim", user_input);

  float sparsity = floatParameter(config, "sparsity", user_input);

  std::string activation = stringParameter(config, "activation", user_input);

  bolt::FullyConnectedNodePtr node;
  if (config.contains("sampling_config")) {
    const auto& sampling_json = config["sampling_config"];

    if (sampling_json.is_string() &&
        sampling_json.get<std::string>() == "random") {
      node = bolt::FullyConnectedNode::make(
          dim, sparsity, activation,
          std::make_shared<bolt::RandomSamplingConfig>());

    } else if (sampling_json.is_object()) {
      uint32_t num_tables =
          integerParameter(sampling_json, "num_tables", user_input);
      uint32_t hashes_per_table =
          integerParameter(sampling_json, "hashes_per_table", user_input);
      uint32_t reservoir_size =
          integerParameter(sampling_json, "reservoir_size", user_input);

      node = bolt::FullyConnectedNode::make(
          dim, sparsity, activation,
          std::make_shared<bolt::DWTASamplingConfig>(
              num_tables, hashes_per_table, reservoir_size));

    } else {
      throw std::invalid_argument(
          "sampling_config must be a string 'random' or an object providing "
          "sampling parameters.");
    }
  } else {
    node = bolt::FullyConnectedNode::makeAutotuned(dim, sparsity, activation);
  }

  std::string predecessor = stringValue(config, "predecessor");
  if (!created_nodes.count(predecessor)) {
    throw std::invalid_argument("Could not find node '" + predecessor + "'.");
  }
  node->addPredecessor(created_nodes.at(predecessor));

  return node;
}

bolt::BoltGraphPtr buildModel(const json& config,
                              const ParameterInputMap& user_input,
                              const std::vector<uint32_t>& input_dims) {
  std::unordered_map<std::string, bolt::NodePtr> created_nodes;

  auto json_inputs = arrayValue(config, "inputs");
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

  for (const auto& node_config : arrayValue(config, "nodes")) {
    if (!node_config.is_object()) {
      throw std::invalid_argument("Node config must be an json object.");
    }

    std::string name = stringValue(node_config, "name");

    std::string type = stringValue(node_config, "type");
    if (type == "fully_connected") {
      created_nodes[name] =
          buildFullyConnectedNode(node_config, user_input, created_nodes);
    } else {
      throw std::invalid_argument("Found unsupported node type '" + type +
                                  "'.");
    }
  }

  std::string output = stringValue(config, "output");
  if (!created_nodes.count(output)) {
    throw std::invalid_argument("Could not find node '" + output + "'.");
  }

  auto loss = bolt::getLossFunction(stringValue(config, "loss"));

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