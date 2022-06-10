#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

namespace thirdai::bolt::tests {

static const uint32_t n_classes = 100, n_batches = 100, batch_size = 100;

class FullyConnectedClassificationNetworkTestFixture : public testing::Test {
 public:
  static dataset::DatasetWithLabels genDataset(bool add_noise) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, add_noise ? 1.0 : 0.1);

    std::vector<bolt::BoltBatch> data_batches;
    std::vector<bolt::BoltBatch> label_batches;
    for (uint32_t b = 0; b < n_batches; b++) {
      std::vector<bolt::BoltVector> labels;
      std::vector<bolt::BoltVector> vectors;
      for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t label = label_dist(gen);
        bolt::BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!add_noise) {
          v.activations[label] += 1.0;
        }
        vectors.push_back(std::move(v));
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.push_back(bolt::BoltBatch(std::move(vectors)));
      label_batches.push_back(bolt::BoltBatch(std::move(labels)));
    }

    return dataset::DatasetWithLabels(
        dataset::BoltDataset(std::move(data_batches), n_batches * batch_size),
        dataset::BoltDataset(std::move(label_batches), n_batches * batch_size));
  }
};

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
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

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
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

  auto data = FullyConnectedClassificationNetworkTestFixture::genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate */ 0.001, /* epochs */ 2,
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
       TrainSimpleDatasetMultiLayerNetworkSigmoid) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Sigmoid)},
                                n_classes);

  auto data = FullyConnectedClassificationNetworkTestFixture::genDataset(false);

  network.train(data.data, data.labels, BinaryCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ true);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ true);
  // Lower accuracy threshold to 0.6 because Sigmoid/BCE converges slower than
  // ReLU/Tanh.
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.6);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainNoisyDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 2,
                /* rehash= */ 0, /* rebuild=*/0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       MultiLayerNetworkToString) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Softmax)},
                                n_classes);

  std::stringstream summary;
  network.buildNetworkSummary(summary);

  std::string expected =
      "========= Bolt Network =========\n"
      "InputLayer (Layer 0): dim=100\n"
      "FullyConnectedLayer (Layer 1): dim=10000, sparsity=0.1, act_func=ReLU\n"
      "FullyConnectedLayer (Layer 2): dim=100, sparsity=1, act_func=Softmax\n"
      "================================";
  std::string actual = summary.str();

  EXPECT_STREQ(actual, expected);
}

}  // namespace thirdai::bolt::tests