#include "TestDatasetGenerators.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/Switch.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <random>

namespace thirdai::bolt::tests {

static constexpr uint32_t n_classes = 100;
static constexpr uint32_t n_batches = 100;
static constexpr uint32_t batch_size = 100;
static constexpr uint32_t n_switch_layers = 10;
static constexpr uint32_t seed = 9824;

auto generateSwitchDataset() {
  auto [vector_data, labels] =
      TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ n_classes, /* n_batches= */ n_batches,
          /* batch_size= */ batch_size, /* noisy_dataset= */ false);

  std::vector<dataset::BoltTokenBatch> token_batches;

  std::uniform_int_distribution<uint32_t> int_dist(0, n_switch_layers - 1);
  std::mt19937 rand(seed);

  for (uint32_t batch_index = 0; batch_index < vector_data->numBatches();
       batch_index++) {
    std::vector<std::vector<uint32_t>> masked_indices;

    for (uint32_t i = 0; i < vector_data->batchSize(batch_index); i++) {
      masked_indices.push_back({int_dist(rand)});
    }

    token_batches.emplace_back(
        dataset::BoltTokenBatch(std::move(masked_indices)));
  }

  auto masked_indices_dataset =
      std::make_shared<dataset::BoltTokenDataset>(std::move(token_batches));

  return std::make_tuple(vector_data, masked_indices_dataset, labels);
}

TEST(SwitchNodeTest, TrainsOnSimpleClassificationDataset) {
  auto [train_data, train_tokens, train_labels] = generateSwitchDataset();

  auto input = std::make_shared<Input>(/* dim= */ n_classes);
  auto token_input = std::make_shared<TokenInput>();

  auto switch_layer = std::make_shared<SwitchNode>(
      /* dim= */ 100, /* activation= */ "relu",
      /* n_layers= */ n_switch_layers);
  switch_layer->addPredecessors(input, token_input);

  auto output = std::make_shared<FullyConnectedNode>(
      /* dim= */ n_classes, /* activation= */ "softmax");
  output->addPredecessor(switch_layer);

  BoltGraph model(/* inputs= */ {input}, /* token_inputs= */ {token_input},
                  /* output= */ output);

  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto train_cfg = TrainConfig::makeConfig(/* learning_rate= */ 0.001, 5);

  model.train({train_data}, {train_tokens}, train_labels, train_cfg);

  auto [test_data, test_tokens, test_labels] = generateSwitchDataset();

  auto predict_cfg =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  auto result =
      model.predict({test_data}, {test_tokens}, test_labels, predict_cfg);

  ASSERT_GE(result.first["categorical_accuracy"], 0.95);
}

}  // namespace thirdai::bolt::tests