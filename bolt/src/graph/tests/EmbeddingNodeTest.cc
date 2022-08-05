#include "TestDatasetGenerators.h"
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

  auto [data, labels] = TestDatasetGenerators::generateSimpleTokenDataset(
      /* n_batches= */ num_batches, /* batch_size= */ batch_size,
      /* seed= */ seed);

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