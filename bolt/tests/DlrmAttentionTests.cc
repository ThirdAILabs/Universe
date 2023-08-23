
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/DlrmAttention.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt/tests/TestDatasetGenerators.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

/**
 * Tests on a task where the first dataset is a list of vectors of dimension
 * n_ids in which one random id is one hot encoded, and a small amound of noise
 * is added to the rest of the vector. The second dataset is a set of tokens
 * which can possibly contain the one hot encoded id. The label is 1 if the set
 * contains the one hot encoded id and 0 otherwise. The intuition is that we are
 * looking for similarity between the tokens and the dense input vector, and
 * this similarity will be captured by the dot product attention in the
 * DlrmAttentionLayer.
 */
TEST(DlrmAttentionTests, TestSetMembership) {
  uint32_t n_ids = 1000, n_tokens = 5, batch_size = 100, n_batches = 100;

  auto dense_input = Input::make(n_ids);

  auto fc_hidden = FullyConnected::make(
                       /* dim= */ 20, /* input_dim= */ dense_input->dim(),
                       /* sparsity= */ 1.0, /* activation= */ "relu")
                       ->apply(dense_input);

  auto token_input = Input::make(/* dim= */ n_ids);

  auto embedding = RobeZ::make(
                       /* num_embedding_lookups */ 4, /* lookup_size= */ 5,
                       /* log_embedding_block_size= */ 14,
                       /* reduction= */ "concatenation",
                       /* num_tokens_per_input= */ n_tokens)
                       ->apply(token_input);

  auto dlrm_attention = DlrmAttention::make()->apply(fc_hidden, embedding);

  auto output = FullyConnected::make(
                    /* dim= */ 2, /* input_dim= */ dlrm_attention->dim(),
                    /* sparsity= */ 1.0, /* activation= */ "softmax")
                    ->apply(dlrm_attention);

  auto labels = Input::make(/* dim= */ 2);
  auto loss = CategoricalCrossEntropy::make(output, labels);

  auto model = Model::make({dense_input, token_input}, {output}, {loss});

  model->summary();

  auto dataset =
      bolt::tests::TestDatasetGenerators::generateDlrmAttentionDataset(
          n_ids, n_tokens, n_batches, batch_size,
          /* seed= */ 24090);

  auto input_dataset = convertDatasets(
      {std::get<0>(dataset), std::get<1>(dataset)}, model->inputDims());
  auto label_dataset = convertDataset(std::get<2>(dataset), /* dim= */ 2);

  Trainer trainer(model);

  auto metrics = trainer.train_with_metric_names(
      /* train_data= */ {input_dataset, label_dataset},
      /* learning_rate= */ 0.001, /* epochs= */ 10, /* train_metrics= */ {},
      /* validation_data= */ {{input_dataset, label_dataset}},
      /* validation_metrics= */ {"categorical_accuracy"});

  ASSERT_GE(metrics.at("val_categorical_accuracy").back(), 0.9);
}

}  // namespace thirdai::bolt::tests