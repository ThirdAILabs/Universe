#include "TestDatasetGenerators.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/DlrmAttention.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <cstdio>

namespace thirdai::bolt::tests {

/**
 * Generates a dataset where the first dataset is a list of vectors of
 * dimensin n_ids in which one random id is one hot encoded, and a small
 * amound of noise is added to the rest of the vector. The second dataset is a
 * set of tokens which can possible contain the one hot encoded id. The label
 * is 1 if the set contains the one hot encoded id and 0 otherwise. The
 * intuition is that we are looking for similarity between the tokens and the
 * dense input vector, and this similarity will be captured by the dot product
 * attention in the DlrmAttentionLayer.
 */
TEST(DlrmAttentionNodeTest, TestSetMembership) {
  uint32_t n_ids = 1000, n_tokens = 5, batch_size = 100, n_batches = 100;

  auto dense_input = Input::make(n_ids);

  auto fc_hidden = FullyConnectedNode::makeDense(
                       /* dim= */ 20, /* activation= */ "relu")
                       ->addPredecessor(dense_input);

  auto token_input = Input::makeTokenInput(
      /* expected_dim= */ n_ids,
      /* num_tokens_range= */ std::pair<uint32_t, uint32_t>(n_tokens,
                                                            n_tokens));

  auto embedding = EmbeddingNode::make(
                       /* num_embedding_lookups */ 4, /* lookup_size= */ 5,
                       /* log_embedding_block_size= */ 14,
                       /* reduction= */ "concatenation",
                       /* num_tokens_per_input= */ n_tokens)
                       ->addInput(token_input);

  auto dlrm_attention = std::make_shared<DlrmAttentionNode>()->setPredecessors(
      fc_hidden, embedding);

  auto output = FullyConnectedNode::makeDense(
                    /* dim= */ 2, /* activation= */ "softmax")
                    ->addPredecessor(dlrm_attention);

  BoltGraph model(/* inputs= */ {dense_input, token_input},
                  /* output= */ output);

  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [data, tokens, labels] =
      TestDatasetGenerators::generateDlrmAttentionDataset(
          n_ids, n_tokens, n_batches, batch_size, /* seed= */ 24090);

  auto train_cfg = TrainConfig::makeConfig(0.001, 10);
  auto eval_cfg =
      EvalConfig::makeConfig().withMetrics({"categorical_accuracy"});

  model.train({data, tokens}, labels, train_cfg);
  auto [metrics, _] = model.evaluate({data, tokens}, labels, eval_cfg);

  ASSERT_GE(metrics["categorical_accuracy"], 0.9);

  // We do a save and load test here, it should ideally be a seperate test
  // but if we do it here we have access to the trained model and data and
  // don't need a helper method that returns a tuple.
  auto original_accuracy = metrics["categorical_accuracy"];

  std::string save_filename = "./tmp_dlrm_attention_model";
  model.save(save_filename);
  auto loaded_model = BoltGraph::load(save_filename);

  auto loaded_accuracy = model.evaluate({data, tokens}, labels, eval_cfg)
                             .first["categorical_accuracy"];
  ASSERT_EQ(original_accuracy, loaded_accuracy);

  std::remove(save_filename.c_str());
}

}  // namespace thirdai::bolt::tests