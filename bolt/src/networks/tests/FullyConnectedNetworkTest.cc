#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <algorithm>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

class FullyConnectedNetworkTestFixture : public testing::Test {
 public:
  static const uint32_t n_classes = 100, n_batches = 100, batch_size = 100;

  static dataset::InMemoryDataset<dataset::BoltInputBatch> genDataset(
      bool add_noise) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, add_noise ? 1.0 : 0.1);

    std::vector<dataset::BoltInputBatch> batches;
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
      batches.push_back(
          dataset::BoltInputBatch(std::move(vectors), std::move(labels)));
    }

    return dataset::InMemoryDataset<dataset::BoltInputBatch>(
        std::move(batches), n_batches * batch_size);
  }
};

TEST_F(FullyConnectedNetworkTestFixture, TrainSimpleDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(false);

  network.train(data, CategoricalCrossEntropyLoss(), 0.001, 5);
  auto test_metrics = network.predict(data, nullptr, {"categorical_accuracy"});
  ASSERT_GE(test_metrics["categorical_accuracy"].front(), 0.99);
}

TEST_F(FullyConnectedNetworkTestFixture, TrainNoisyDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data, CategoricalCrossEntropyLoss(), 0.001, 5);
  auto test_metrics = network.predict(data, nullptr, {"categorical_accuracy"});
  ASSERT_LE(test_metrics["categorical_accuracy"].front(), 0.2);
}

TEST_F(FullyConnectedNetworkTestFixture, TrainSimpleDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           10000, 0.1, ActivationFunction::ReLU, SamplingConfig(3, 32, 9, 32)),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = genDataset(false);

  network.train(data, CategoricalCrossEntropyLoss(), 0.001, 2);
  auto test_metrics = network.predict(data, nullptr, {"categorical_accuracy"});
  ASSERT_GE(test_metrics["categorical_accuracy"].front(), 0.99);
}

TEST_F(FullyConnectedNetworkTestFixture, TrainNoisyDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           10000, 0.1, ActivationFunction::ReLU, SamplingConfig(3, 32, 9, 32)),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = genDataset(true);

  network.train(data, CategoricalCrossEntropyLoss(), 0.001, 2);
  auto test_metrics = network.predict(data, nullptr, {"categorical_accuracy"});
  ASSERT_LE(test_metrics["categorical_accuracy"].front(), 0.2);
}

}  // namespace thirdai::bolt::tests