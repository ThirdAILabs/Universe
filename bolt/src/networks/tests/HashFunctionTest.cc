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

static const uint32_t n_classes = 100, n_batches = 100, batch_size = 100;

class HashFunctionTestFixture : public testing::Test {
 public:
  static dataset::DatasetWithLabels genDataset(bool input_is_noise) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, input_is_noise ? 1.0 : 0.1);

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
        if (!input_is_noise) {
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

static void testSimpleDatasetHashFunction(HashingFunction hash_function) {
  // constructs the network with user provided hash function.
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           10000, 0.1, ActivationFunction::ReLU,
           SamplingConfig(5, 64, 15, 4, hash_function)),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = HashFunctionTestFixture::genDataset(false);

  // train the network for two epochs
  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto first_test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);

  // train the network for 5 epochs
  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(), 0.001, 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto second_test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);

  // assert that the accuracy improves.
  ASSERT_GE(second_test_metrics["categorical_accuracy"],
            first_test_metrics["categorical_accuracy"]);
}

// test for DWTA Hash Function
TEST_F(HashFunctionTestFixture, TrainSimpleDatasetDWTA) {
  testSimpleDatasetHashFunction(HashingFunction::DWTA);
}

// test for SRP Hash Function
TEST_F(HashFunctionTestFixture, TrainSimpleDatasetSRP) {
  testSimpleDatasetHashFunction(HashingFunction::SRP);
}

// test for FastSRP Hash Function
TEST_F(HashFunctionTestFixture, TrainSimpleDatasetFastSRP) {
  testSimpleDatasetHashFunction(HashingFunction::FastSRP);
}

}  // namespace thirdai::bolt::tests