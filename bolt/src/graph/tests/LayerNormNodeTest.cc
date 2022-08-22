
#include "TestDatasetGenerators.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/layers/LayerConfig.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>

namespace thirdai::bolt::tests {

static constexpr uint32_t n_classes = 100;
static constexpr uint32_t batch_size = 32;
static constexpr uint32_t n_batches = 100;
static constexpr uint32_t epochs = 5;
static constexpr float accuracy_threshold = 0.95;

NormalizationLayerConfig getLayerNormConfig() {
  return NormalizationLayerConfig::makeConfig()
      .setCenteringFactor(/*centering_factor= */ 0.0)
      .setScalingFactor(/* scaling_factor= */ 1.0);
}

static BoltGraph buildSingleNormNodeModel() {
  auto input = std::make_shared<Input>(/* expected_input_dim */ n_classes);
  auto hidden_layer = std::make_shared<FullyConnectedNode>(2000, "relu");
  hidden_layer->addPredecessor(input);

  NormalizationLayerConfig layer_norm_config = getLayerNormConfig();

  auto hidden_norm_layer = std::make_shared<LayerNormNode>(layer_norm_config);
  hidden_norm_layer->addPredecessor(hidden_layer);

  auto output = std::make_shared<FullyConnectedNode>(
      /* expected_dim */ n_classes, "softmax");
  output->addPredecessor(hidden_norm_layer);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return model;
}

void testLayerNormNodeForwardAndBackwardPass() {
  auto [data, labels] = TestDatasetGenerators::generateSimpleVectorDataset(
      /* n_classes= */ n_classes, /* n_batches= */ n_batches,
      /* batch_size= */ batch_size, /* noisy_dataset= */ false);

  BoltGraph model = buildSingleNormNodeModel();

  auto train_config = TrainConfig::makeConfig(/* learning_rate= */ 0.001,
                                              /* epochs= */ epochs);

  model.train({data}, {}, labels, train_config);

  auto predict_config =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  auto test_metrics = model.predict({data}, {}, labels, predict_config).first;

  ASSERT_GE(test_metrics["categorical_accuracy"], accuracy_threshold);
}

TEST(LayerNormNodeTest, HiddenLayerNormalizationTest) {
  testLayerNormNodeForwardAndBackwardPass();
}

}  // namespace thirdai::bolt::tests
