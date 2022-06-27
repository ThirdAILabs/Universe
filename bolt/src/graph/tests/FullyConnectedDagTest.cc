#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

namespace thirdai::bolt::tests {

static BoltGraph getSingleLayerModel() {
  auto input_layer = std::make_shared<Input>(n_classes);
  auto output_layer = std::make_shared<FullyConnectedLayerNode>(
      n_classes, ActivationFunction::Softmax);

  output_layer->addPredecessor(input_layer);

  BoltGraph model({input_layer}, output_layer);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return model;
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetSingleLayerNetwork) {
  BoltGraph model = getSingleLayerModel();

  auto data = genDataset(false);

  model.train(data.data, data.labels,
              /* learning_rate= */ 0.001, /* epochs= */ 5,
              /* rebuild_hash_tables= */ 0, /* reconstruct_hash_functions= */ 0,
              /* metric_names= */ {},
              /* verbose= */ false);
  auto test_metrics =
      model.predict(data.data, data.labels,
                    /* metric_names= */ {"categorical_accuracy"},
                    /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.98);
}

TEST(FullyConnectedDagTest, TrainNoisyDatasetSingleLayerNetwork) {
  BoltGraph model = getSingleLayerModel();

  auto data = genDataset(true);

  model.train(data.data, data.labels,
              /* learning_rate= */ 0.001, /* epochs= */ 5,
              /* rebuild_hash_tables= */ std::nullopt,
              /* reconstruct_hash_functions= */ std::nullopt,
              /* metric_names= */ {},
              /* verbose= */ false);
  auto test_metrics =
      model.predict(data.data, data.labels,
                    /* metric_names= */ {"categorical_accuracy"},
                    /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

static BoltGraph getMultiLayerModel(ActivationFunction hidden_layer_act,
                                    ActivationFunction output_layer_act) {
  auto input_layer = std::make_shared<Input>(n_classes);

  auto hidden_layer =
      std::make_shared<FullyConnectedLayerNode>(10000, 0.1, hidden_layer_act);

  hidden_layer->addPredecessor(input_layer);

  auto output_layer =
      std::make_shared<FullyConnectedLayerNode>(n_classes, output_layer_act);

  output_layer->addPredecessor(hidden_layer);

  BoltGraph model({input_layer}, output_layer);

  EXPECT_TRUE(output_layer_act == ActivationFunction::Softmax ||
              output_layer_act == ActivationFunction::Sigmoid);

  if (output_layer_act == ActivationFunction::Softmax) {
    model.compile(std::make_shared<CategoricalCrossEntropyLoss>());
  } else if (output_layer_act == ActivationFunction::Sigmoid) {
    model.compile(std::make_shared<BinaryCrossEntropyLoss>());
  }

  return model;
}

static void testSimpleDatasetMultiLayerModel(
    ActivationFunction hidden_layer_act, ActivationFunction output_layer_act,
    uint32_t epochs) {
  BoltGraph model = getMultiLayerModel(hidden_layer_act, output_layer_act);

  auto data = genDataset(false);

  auto train_metrics =
      model.train(data.data, data.labels,
                  /* learning_rate */ 0.001, /* epochs */ epochs,
                  /* rebuild_hash_tables= */ std::nullopt,
                  /* reconstruct_hash_functions= */ std::nullopt,
                  /* metric_names= */ {"mean_squared_error"},
                  /* verbose= */ false);
  ASSERT_LT(train_metrics.at("mean_squared_error").back(),
            train_metrics.at("mean_squared_error").front());

  auto test_metrics =
      model.predict(data.data, data.labels,
                    /* metric_names= */ {"categorical_accuracy"},
                    /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkRelu) {
  testSimpleDatasetMultiLayerModel(ActivationFunction::ReLU,
                                   ActivationFunction::Softmax, 2);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkTanh) {
  testSimpleDatasetMultiLayerModel(ActivationFunction::Tanh,
                                   ActivationFunction::Softmax, 2);
}

TEST(FullyConnectedDagTest, TrainSimpleDatasetMultiLayerNetworkSigmoid) {
  testSimpleDatasetMultiLayerModel(ActivationFunction::ReLU,
                                   ActivationFunction::Sigmoid, 5);
}

TEST(FullyConnectedDagTest, TrainNoisyDatasetMultiLayerNetwork) {
  BoltGraph model =
      getMultiLayerModel(ActivationFunction::ReLU, ActivationFunction::Softmax);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto data = genDataset(true);

  model.train(data.data, data.labels,
              /* learning_rate= */ 0.001, /* epochs= */ 2,
              /* rebuild_hash_tables= */ std::nullopt,
              /* reconstruct_hash_functions=*/std::nullopt,
              /* metric_names= */ {},
              /* verbose= */ false);

  auto test_metrics =
      model.predict(data.data, data.labels,
                    /* metric_names= */ {"categorical_accuracy"},
                    /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

}  // namespace thirdai::bolt::tests