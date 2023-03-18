#include "Models.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/udt/Defaults.h>

namespace thirdai::automl::udt::utils {

bolt::BoltGraphPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                              const config::ArgumentMap& args,
                              const std::optional<std::string>& model_config,
                              bool use_sigmoid_bce) {
  if (model_config) {
    return utils::loadModel({input_dim}, output_dim, *model_config);
  }
  uint32_t hidden_dim = args.get<uint32_t>("embedding_dimension", "integer",
                                           defaults::HIDDEN_DIM);
  return utils::defaultModel(input_dim, hidden_dim, output_dim,
                             use_sigmoid_bce);
}

namespace {

float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},    {900, 0.2},    {1800, 0.1},     {4000, 0.05},
      {10000, 0.02}, {20000, 0.01}, {1000000, 0.005}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return sparsity_values.back().second;
}

}  // namespace

bolt::BoltGraphPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                uint32_t output_dim, bool use_sigmoid_bce) {
  bolt::InputPtr input_node = bolt::Input::make(input_dim);

  auto hidden = bolt::FullyConnectedNode::makeDense(hidden_dim,
                                                    /* activation= */ "relu");
  hidden->addPredecessor(input_node);

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = use_sigmoid_bce ? "sigmoid" : "softmax";
  auto output =
      bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity, activation);
  output->addPredecessor(hidden);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ std::vector<bolt::InputPtr>{input_node}, output);

  use_sigmoid_bce
      ? graph->compile(
            bolt::BinaryCrossEntropyLoss::makeBinaryCrossEntropyLoss())
      : graph->compile(bolt::CategoricalCrossEntropyLoss::
                           makeCategoricalCrossEntropyLoss());

  return graph;
}

bolt::BoltGraphPtr loadModel(const std::vector<uint32_t>& input_dims,
                             uint32_t output_dim,
                             const std::string& config_path) {
  config::ArgumentMap parameters;
  parameters.insert("output_dim", output_dim);

  auto json_config = json::parse(config::loadConfig(config_path));

  return config::buildModel(json_config, parameters, input_dims);
}

bool hasSoftmaxOutput(const bolt::BoltGraphPtr& model) {
  auto fc_output =
      std::dynamic_pointer_cast<bolt::FullyConnectedNode>(model->output());
  return fc_output && (fc_output->getActivationFunction() ==
                       bolt::ActivationFunction::Softmax);
}

}  // namespace thirdai::automl::udt::utils