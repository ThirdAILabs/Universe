#include "UDTDefault.h"
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/config/ModelConfig.h>

namespace thirdai::automl::nn {

inline float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},   {900, 0.2},    {1800, 0.1},
      {4000, 0.05}, {10000, 0.02}, {20000, 0.01}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return 0.05;
}

bolt::BoltGraphPtr UDTDefault(const std::vector<uint32_t>& input_dims,
                              uint32_t output_dim, uint32_t hidden_layer_size) {
  auto hidden = bolt::FullyConnectedNode::makeDense(hidden_layer_size,
                                                    /* activation= */ "relu");

  std::vector<bolt::InputPtr> input_nodes;
  input_nodes.reserve(input_dims.size());
  for (uint32_t input_dim : input_dims) {
    input_nodes.push_back(bolt::Input::make(input_dim));
  }

  hidden->addPredecessor(input_nodes[0]);

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = "softmax";
  auto output =
      bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity, activation);
  output->addPredecessor(hidden);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ input_nodes, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

bolt::BoltGraphPtr fromConfig(const std::vector<uint32_t>& input_dims,
                              uint32_t output_dim,
                              const std::string& saved_model_config) {
  // This will pass the output (label) dimension of the model into the model
  // config so that it can be used to determine the model architecture.

  config::ArgumentMap parameters;
  parameters.insert("output_dim", output_dim);

  auto json_config = json::parse(config::loadConfig(saved_model_config));

  return config::buildModel(json_config, parameters, input_dims);
}

}  // namespace thirdai::automl::nn