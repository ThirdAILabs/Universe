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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::config {

bolt::NodePtr buildFullyConnectedNode(
    const json& config, const ParameterInputMap& user_input,
    const std::unordered_map<std::string, bolt::NodePtr>& created_nodes) {
  uint32_t dim = parameter::integer(config, "dim", user_input);
  float sparsity = parameter::decimal(config, "sparsity", user_input);
  std::string activation = parameter::str(config, "activation", user_input);

  bolt::FullyConnectedNodePtr node;
  if (config.contains("sampling_config")) {
    const auto& sampling_json = config["sampling_config"];
    if (sampling_json.is_string() &&
        sampling_json.get<std::string>() == "random") {
      node = bolt::FullyConnectedNode::make(
          dim, sparsity, activation,
          std::make_shared<bolt::RandomSamplingConfig>());
    } else {
      uint32_t num_tables =
          parameter::integer(sampling_json, "num_tables", user_input);
      uint32_t hashes_per_table =
          parameter::integer(sampling_json, "hashes_per_table", user_input);
      uint32_t reservoir_size =
          parameter::integer(sampling_json, "reservoir_size", user_input);

      node = bolt::FullyConnectedNode::make(
          dim, sparsity, activation,
          std::make_shared<bolt::DWTASamplingConfig>(
              num_tables, hashes_per_table, reservoir_size));
    }
  } else {
    node = bolt::FullyConnectedNode::makeAutotuned(dim, sparsity, activation);
  }

  std::string predecessor = config["predecessor"].get<std::string>();
  if (!created_nodes.count(predecessor)) {
    throw std::invalid_argument("Could not find node '" + predecessor + "'.");
  }
  node->addPredecessor(created_nodes.at(predecessor));

  return node;
}

bolt::BoltGraphPtr buildModel(
    const json& config, const ParameterInputMap& user_input,
    const std::unordered_map<std::string, uint32_t>& input_dims) {
  std::unordered_map<std::string, bolt::NodePtr> created_nodes;

  std::vector<bolt::InputPtr> inputs;
  for (const auto& [name, dim] : input_dims) {
    inputs.push_back(bolt::Input::make(dim));
    created_nodes[name] = inputs.back();
  }

  for (const auto& [name, node_config] : config["nodes"].items()) {
    std::string type = node_config["type"].get<std::string>();
    if (type == "fully_connected") {
      created_nodes[name] =
          buildFullyConnectedNode(node_config, user_input, created_nodes);
    } else {
      throw std::invalid_argument("Found unsupported node type '" + type +
                                  "'.");
    }
  }

  std::string output = config["output"].get<std::string>();
  if (!created_nodes.count(output)) {
    throw std::invalid_argument("Could not find node '" + output + "'.");
  }

  auto loss = bolt::getLossFunction(config["loss"].get<std::string>());

  auto model =
      std::make_shared<bolt::BoltGraph>(inputs, created_nodes.at(output));

  model->compile(loss);

  return model;
}

void dumpConfig(const std::string& config, const std::string& filename) {
  std::ofstream file(filename);
  file << config;
}

std::string loadConfig(const std::string& filename) {
  std::ifstream file(filename);

  std::stringstream buffer;
  buffer << file.rdbuf();

  return buffer.str();
}

}  // namespace thirdai::automl::config