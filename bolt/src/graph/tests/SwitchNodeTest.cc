#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/Switch.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <dataset/src/batch_types/MaskedSentenceBatch.h>
#include <random>

namespace thirdai::bolt::tests {

static uint32_t n_classes = 100;
static uint32_t n_switch_layers = 10;
static uint32_t seed = 9824;

auto getMLMDataset() {
  auto [data, labels] = genDataset(n_classes, /* noisy_dataset= */ false);

  std::vector<dataset::MaskedSentenceBatch> batches;

  std::uniform_int_distribution<uint32_t> int_dist(0, n_switch_layers - 1);
  std::mt19937 rand(seed);

  for (uint32_t batch_index = 0; batch_index < data->numBatches();
       batch_index++) {
    std::vector<std::vector<uint32_t>> masked_indices;
    BoltBatch& vec_batch = data->at(batch_index);

    for (uint32_t i = 0; i < vec_batch.getBatchSize(); i++) {
      masked_indices.push_back({int_dist(rand)});
    }

    batches.emplace_back(std::move(vec_batch),
                         dataset::BoltTokenBatch(std::move(masked_indices)));
  }

  auto dataset =
      std::make_shared<dataset::InMemoryDataset<dataset::MaskedSentenceBatch>>(
          std::move(batches), labels->len());

  return std::make_pair(dataset, labels);
}

TEST(SwitchNodeTest, TrainsOnSimpleClassificationDataset) {
  auto [data, labels] = getMLMDataset();

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

  model.train(data, labels, train_cfg);

  auto predict_cfg =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  auto result = model.predict(data, labels, predict_cfg);

  ASSERT_GE(result.first["categorical_accuracy"], 0.95);
}

}  // namespace thirdai::bolt::tests