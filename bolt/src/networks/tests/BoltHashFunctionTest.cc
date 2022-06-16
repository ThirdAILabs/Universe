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

class BoltHashFunctionTestFixture : public testing::Test {};

static void testSimpleDatasetHashFunction(const std::string& hash_function) {
  // As we train for more epochs, the model should learn better using these hash
  // functions.
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           /*dim = */ 10000, /*sparsity = */ 0.1,
           /*act_func = */ ActivationFunction::ReLU,
           /*sampling_config = */
           SamplingConfig(/*hashes_per_table = */ 5, /*num_tables = */ 64,
                          /*range_pow = */ 15, /*reservoir size = */ 4,
                          /*hash_function = */ hash_function)),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = genDataset(/*add_noise = */ false);

  // train the network for two epochs
  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /*learning_rate = */ 0.001, /*epochs = */ 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto first_test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);

  // train the network for 5 epochs
  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /*learning_rate = */ 0.001, /*epochs = */ 5,
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
TEST_F(BoltHashFunctionTestFixture, TrainSimpleDatasetDWTA) {
  testSimpleDatasetHashFunction("DWTA");
}

// test for SRP Hash Function
TEST_F(BoltHashFunctionTestFixture, TrainSimpleDatasetSRP) {
  testSimpleDatasetHashFunction("SRP");
}

// test for FastSRP Hash Function
TEST_F(BoltHashFunctionTestFixture, TrainSimpleDatasetFastSRP) {
  testSimpleDatasetHashFunction("FastSRP");
}

}  // namespace thirdai::bolt::tests