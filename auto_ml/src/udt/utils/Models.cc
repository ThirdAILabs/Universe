#include "Models.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>

namespace thirdai::automl::udt::utils {

namespace {

float autotuneSparsity(uint32_t dim) {
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

}  // namespace

bolt::BoltGraphPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                uint32_t output_dim) {
  bolt::InputPtr input_node = bolt::Input::make(input_dim);

  auto hidden = bolt::FullyConnectedNode::makeDense(hidden_dim,
                                                    /* activation= */ "relu");
  hidden->addPredecessor(input_node);

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = "softmax";
  auto output =
      bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity, activation);
  output->addPredecessor(hidden);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ std::vector<bolt::InputPtr>{input_node}, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

}  // namespace thirdai::automl::udt::utils