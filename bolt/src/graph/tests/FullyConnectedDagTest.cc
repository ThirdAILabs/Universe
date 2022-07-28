#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

namespace thirdai::bolt::tests {

uint32_t n_classes = 100;

static BoltGraph getSingleLayerModel() {
  auto input_layer = std::make_shared<Input>(n_classes);
  auto output_layer =
      std::make_shared<FullyConnectedNode>(n_classes, "softmax");

  output_layer->addPredecessor(input_layer);

  BoltGraph model({input_layer}, output_layer);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return model;
}

static TrainConfig getTrainConfig(uint32_t epochs) {
  TrainConfig config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ epochs)
          .withMetrics({"mean_squared_error"})
          .silence();

  return config;
}

static PredictConfig getPredictConfig() {
  PredictConfig config = PredictConfig::makeConfig()
                             .withMetrics({"categorical_accuracy"})
                             .silence();

  return config;
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetSingleLayerNetwork) {
  BoltGraph model = getSingleLayerModel();

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);

  model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
              getTrainConfig(/* epochs= */ 5));

  auto test_metrics =
      model.predict(/* test_data= */ {data}, /* test_tokens= */ {}, labels,
                    getPredictConfig());

  ASSERT_GE(test_metrics.first["categorical_accuracy"], 0.98);
}

TEST(FullyConnectedDagTest, TrainNoisyDatasetSingleLayerNetwork) {
  BoltGraph model = getSingleLayerModel();

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ true);

  model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
              getTrainConfig(/* epochs= */ 5));

  auto test_metrics =
      model.predict(/* test_data= */ {data}, /* test_tokens= */ {}, labels,
                    getPredictConfig());

  ASSERT_LE(test_metrics.first["categorical_accuracy"], 0.2);
}

static BoltGraph getMultiLayerModel(const std::string& hidden_layer_act,
                                    const std::string& output_layer_act) {
  auto input_layer = std::make_shared<Input>(n_classes);

  auto hidden_layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 10000, /* sparsity= */ 0.1,
      /* activation= */ hidden_layer_act);

  hidden_layer->addPredecessor(input_layer);

  auto output_layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ n_classes, /* activation= */ output_layer_act);

  output_layer->addPredecessor(hidden_layer);

  BoltGraph model({input_layer}, output_layer);

  EXPECT_TRUE(output_layer_act == "softmax" || output_layer_act == "sigmoid");

  if (output_layer_act == "softmax") {
    model.compile(std::make_shared<CategoricalCrossEntropyLoss>());
  } else if (output_layer_act == "sigmoid") {
    model.compile(std::make_shared<BinaryCrossEntropyLoss>());
  }

  return model;
}

static void testSimpleDatasetMultiLayerModel(
    const std::string& hidden_layer_act, const std::string& output_layer_act,
    uint32_t epochs) {
  BoltGraph model = getMultiLayerModel(hidden_layer_act, output_layer_act);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);

  auto train_metrics =
      model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
                  getTrainConfig(epochs));

  ASSERT_LT(train_metrics.at("mean_squared_error").back(),
            train_metrics.at("mean_squared_error").front());

  auto test_metrics =
      model.predict(/* test_data= */ {data}, /* test_tokens= */ {}, labels,
                    getPredictConfig());

  ASSERT_GE(test_metrics.first["categorical_accuracy"], 0.99);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkRelu) {
  testSimpleDatasetMultiLayerModel("relu", "softmax", /* epochs= */ 2);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkTanh) {
  testSimpleDatasetMultiLayerModel("tanh", "softmax", /* epochs= */ 2);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkSigmoid) {
  testSimpleDatasetMultiLayerModel("relu", "sigmoid", /* epochs= */ 5);
}

TEST(FullyConnectedDagTest, TrainNoisyDatasetMultiLayerNetwork) {
  BoltGraph model = getMultiLayerModel("relu", "softmax");

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ true);

  model.train(/* train_data= */ {data}, /* train_tokens= */ {}, labels,
              getTrainConfig(/* epochs= */ 2));

  auto test_metrics =
      model.predict(/* test_data= */ {data}, /* test_tokens= */ {}, labels,
                    getPredictConfig());

  ASSERT_LE(test_metrics.first["categorical_accuracy"], 0.2);
}

}  // namespace thirdai::bolt::tests