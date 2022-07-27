#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <algorithm>
#include <numeric>
#include <random>

namespace thirdai::bolt::tests {

static constexpr uint32_t num_batches = 100;
static constexpr uint32_t batch_size = 100;
static constexpr uint32_t seed = 24902;

// This generates a dataset of random numbers whose label is 0 if even and 1 if
// odd. This tests the embedding layers ability to essentially memorize the
// dataset.
std::pair<std::shared_ptr<dataset::InMemoryDataset<dataset::BoltTokenBatch>>,
          dataset::BoltDatasetPtr>
genDataset() {
  uint32_t dataset_size = num_batches * batch_size;

  std::vector<uint32_t> tokens(dataset_size);

  std::iota(tokens.begin(), tokens.end(), 1);
  std::shuffle(tokens.begin(), tokens.end(), std::mt19937(seed));

  std::vector<dataset::BoltTokenBatch> data;
  std::vector<BoltBatch> labels;

  for (uint32_t batch_index = 0; batch_index < num_batches; batch_index++) {
    std::vector<std::vector<uint32_t>> batch_data;
    std::vector<BoltVector> batch_labels;

    for (uint32_t vec_index = 0; vec_index < batch_size; vec_index++) {
      uint32_t token = tokens[batch_index * batch_size + vec_index];
      batch_data.push_back({token});
      batch_labels.push_back(BoltVector::makeSparseVector({token % 2}, {1.0}));
    }

    data.push_back(dataset::BoltTokenBatch(std::move(batch_data)));
    labels.push_back(BoltBatch(std::move(batch_labels)));
  }

  return std::make_pair(
      std::make_shared<dataset::InMemoryDataset<dataset::BoltTokenBatch>>(
          std::move(data)),
      std::make_shared<dataset::BoltDataset>(std::move(labels)));
}

TEST(EmbeddingNodeTest, SimpleTokenDataset) {
  auto token_input = std::make_shared<TokenInput>();

  auto embedding_layer = std::make_shared<EmbeddingNode>(
      /* num_embedding_lookups= */ 4, /* lookup_size= */ 8,
      /* log_embedding_block_size= */ 14);
  embedding_layer->addInput(token_input);

  auto fully_connected_layer = std::make_shared<FullyConnectedNode>(
      /* dim= */ 2,
      /* activation= */ "softmax");
  fully_connected_layer->addPredecessor(embedding_layer);

  BoltGraph model(/* inputs= */ {}, /* token_inputs= */ {token_input},
                  /* output= */ fully_connected_layer);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [data, labels] = genDataset();

  TrainConfig train_config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ 10)
          .withMetrics({"mean_squared_error"})
          .silence();

  PredictConfig predict_config = PredictConfig::makeConfig()
                                     .withMetrics({"categorical_accuracy"})
                                     .silence();

  auto train_metrics = model.train(
      /* train_data= */ {}, /* train_tokens= */ {data}, labels, train_config);

  ASSERT_GT(train_metrics["mean_squared_error"].front(),
            train_metrics["mean_squared_error"].back());

  auto test_metrics = model.predict(
      /* test_data= */ {}, /* test_tokens= */ {data}, labels, predict_config);

  ASSERT_GT(test_metrics.first["categorical_accuracy"], 0.9);
}

}  // namespace thirdai::bolt::tests