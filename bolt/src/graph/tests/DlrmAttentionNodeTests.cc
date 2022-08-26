#include "TestDatasetGenerators.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/DlrmAttention.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt::tests {

TEST(DlrmAttentionNodeTest, TestSetMembership) {
  uint32_t n_ids = 1000, n_tokens = 5, batch_size = 100, n_batches = 100;

  auto dense_input = std::make_shared<Input>(n_ids);

  auto fc_hidden = std::make_shared<FullyConnectedNode>(
                       /* dim= */ 20, /* activation= */ "relu")
                       ->addPredecessor(dense_input);

  auto token_input = std::make_shared<TokenInput>();

  auto embedding = std::make_shared<EmbeddingNode>(
                       /* num_embedding_lookups */ 4, /* lookup_size= */ 5,
                       /* log_embedding_block_size= */ 14,
                       /* reduction= */ EmbeddingReductionType::CONCATENATION,
                       /* num_tokens_per_input= */ n_tokens)
                       ->addInput(token_input);

  auto dlrm_attention = std::make_shared<DlrmAttentionNode>()->setPredecessors(
      fc_hidden, embedding);

  auto output = std::make_shared<FullyConnectedNode>(
                    /* dim= */ 2, /* activation= */ "softmax")
                    ->addPredecessor(dlrm_attention);

  BoltGraph model(/* inputs= */ {dense_input},
                  /* token_inputs= */ {token_input}, /* output= */ output);

  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [data, tokens, labels] =
      TestDatasetGenerators::generateDlrmAttentionDataset(
          n_ids, n_tokens, n_batches, batch_size, /* seed= */ 24090);

  auto train_cfg = TrainConfig::makeConfig(0.001, 10);
  auto predict_cfg =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  model.train({data}, {tokens}, labels, train_cfg);
  auto [metrics, _] = model.predict({data}, {tokens}, labels, predict_cfg);

  ASSERT_GE(metrics["categorical_accuracy"], 0.9);
}

}  // namespace thirdai::bolt::tests