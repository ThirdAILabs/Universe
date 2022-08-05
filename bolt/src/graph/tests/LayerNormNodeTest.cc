
#include "MockNode.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace thirdai::bolt::tests {

static uint32_t n_classes = 100;
static uint32_t batch_size = 32;

using NodeGraphTuple = std::tuple<std::shared_ptr<LayerNormNode>, BoltGraph>;

static NodeGraphTuple buildSingleNormNodeModel() {
  auto input = std::make_shared<Input>(/* expected_input_dim */ n_classes);
  auto hidden_layer = std::make_shared<FullyConnectedNode>(2000, 1.0, "ReLU");
  hidden_layer->addPredecessor(input);

  NormalizationLayerConfig layer_norm_config =
      NormalizationLayerConfig::makeConfig();
  auto normalization_layer = std::make_shared<LayerNormNode>(layer_norm_config);
  normalization_layer->addPredecessor(hidden_layer);

  auto output = std::make_shared<FullyConnectedNode>(
      /* expected_dim */ n_classes, "Softmax");
  output->addPredecessor(normalization_layer);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return std::make_tuple(normalization_layer, model);
}

void testLayerNormNodeForwardAndBackwardPass() {
  auto [layer_norm_node, model] = buildSingleNormNodeModel();

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);
  TrainConfig train_config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ 5)
          .withMetrics({"mean_squared_error"})
          .withBatchSize(batch_size)
          .silence();

  model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
              train_config);

  auto pred_node = model.getNodeByName("layer_norm_1")->getPredecessors()[0];
  ASSERT_EQ(pred_node->outputDim(), model.getNodeByName("layer_norm_1")->outputDim());

  BoltVector& output_vector =
      model.getNodeByName("layer_norm_1")->getOutputVector(/* vec_index= */ 1);

  for (uint32_t neuron_index = 0; neuron_index < output_vector.len;
       neuron_index++) {
    ASSERT_NE(output_vector.gradients[neuron_index], 0.0);
  }
}

TEST(LayerNormNodeTest, LayerNormalizationTest) {
  testLayerNormNodeForwardAndBackwardPass();
}

}  // namespace thirdai::bolt::tests
