#include "BoltNetworkTestUtils.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <algorithm>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

class FullyConnectedClassificationNetworkTestFixture : public testing::Test {};

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.98);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainNoisyDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

static void testSimpleDatasetMultiLayerNetworkActivation(
    ActivationFunction act) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(10000, 0.1, act),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetwork) {
  testSimpleDatasetMultiLayerNetworkActivation(ActivationFunction::ReLU);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetworkTanh) {
  testSimpleDatasetMultiLayerNetworkActivation(ActivationFunction::Tanh);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainNoisyDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 2,
                /* rehash= */ 0, /* rebuild=*/0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

}  // namespace thirdai::bolt::tests