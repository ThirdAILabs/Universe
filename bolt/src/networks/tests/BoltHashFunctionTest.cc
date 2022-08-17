#include <bolt/src/graph/tests/TestDatasetGenerators.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

static constexpr uint32_t n_classes = 100;
static constexpr uint32_t n_batches = 100;
static constexpr uint32_t batch_size = 100;

static void testSimpleDatasetHashFunction(
    const SamplingConfigPtr& sampling_config) {
  // As we train for more epochs, the model should learn better using these hash
  // functions.
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           /*dim = */ 10000, /*sparsity = */ 0.1,
           /*act_func = */ "relu",
           /*sampling_config = */
           sampling_config),
       std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  auto [data, labels] = TestDatasetGenerators::generateSimpleVectorDataset(
      /* n_classes= */ n_classes, /* n_batches= */ n_batches,
      /* batch_size= */ batch_size, /* noisy_dataset= */ false);

  // train the network for two epochs
  network.train(data, labels, CategoricalCrossEntropyLoss(),
                /*learning_rate = */ 0.001, /*epochs = */ 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto first_test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ false);

  // train the network for 5 epochs
  network.train(data, labels, CategoricalCrossEntropyLoss(),
                /*learning_rate = */ 0.001, /*epochs = */ 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto second_test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ false);

  // assert that the accuracy improves.
  ASSERT_GE(second_test_metrics["categorical_accuracy"],
            first_test_metrics["categorical_accuracy"]);
}

// test for DWTA Hash Function
TEST(BoltHashFunctionTest, TrainSimpleDatasetDWTA) {
  auto sampling_config = std::make_shared<DWTASamplingConfig>(
      /* num_tables= */ 64, /* hashes_per_table= */ 3,
      /* reservoir_size= */ 32);

  testSimpleDatasetHashFunction(sampling_config);
}

// test for FastSRP Hash Function
TEST(BoltHashFunctionTest, TrainSimpleDatasetFastSRP) {
  auto sampling_config = std::make_shared<FastSRPSamplingConfig>(
      /* num_tables= */ 64, /* hashes_per_table= */ 9,
      /* reservoir_size= */ 32);

  testSimpleDatasetHashFunction(sampling_config);
}

}  // namespace thirdai::bolt::tests