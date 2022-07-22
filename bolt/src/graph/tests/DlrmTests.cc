#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>
#include <tuple>

namespace thirdai::bolt::tests {

/**
 * This test creates a dag model with the following architecture
 *
 *          input         token_input
 *            |                |
 *          fc_layer      embedding_layer
 *             \              /
 *              \            /
 *             concatenate_layer
 *                    |
 *                 fc_layer
 *
 * The vector input is simply the one-hot encoding of the label with some noise
 * added, the token_input is just the label itself. There are 4 tests that are
 * run. The first is with the inputs described above, the second makes token
 * input a random value instead of the label, the third makes the vector input
 * only noise with no one hot encoding for the label, and the final test makes
 * both the vector and token input noisy. The tests then check that the model is
 * able to achieve a reasonable accuracy in the first 3 cases, but not in the
 * fourth. The purpose of this is to check that the model is able to learn from
 * both sets of features.
 */
constexpr uint32_t n_classes = 100;
constexpr uint32_t n_batches = 100;
constexpr uint32_t batch_size = 100;

static std::tuple<dataset::BoltDatasetPtr, dataset::BoltTokenDatasetPtr,
                  dataset::BoltDatasetPtr>
genDlrmDataset(bool noisy_dense_features, bool noisy_categorical_features) {
  std::mt19937 gen(892734);
  std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
  std::normal_distribution<float> data_dist(0,
                                            noisy_dense_features ? 1.0 : 0.1);

  std::vector<BoltBatch> data_batches;
  std::vector<dataset::BoltTokenBatch> token_batches;
  std::vector<BoltBatch> label_batches;
  for (uint32_t b = 0; b < n_batches; b++) {
    std::vector<BoltVector> labels;
    std::vector<BoltVector> dense_features;
    std::vector<std::vector<uint32_t>> categorical_features;
    for (uint32_t i = 0; i < batch_size; i++) {
      uint32_t label = label_dist(gen);
      BoltVector v(n_classes, true, false);
      std::generate(v.activations, v.activations + n_classes,
                    [&]() { return data_dist(gen); });
      if (!noisy_dense_features) {
        v.activations[label] += 1.0;
      }
      dense_features.push_back(std::move(v));
      categorical_features.push_back(
          {noisy_categorical_features ? label + label_dist(gen) : label});
      labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
    }
    data_batches.emplace_back(std::move(dense_features));
    token_batches.emplace_back(std::move(categorical_features));
    label_batches.emplace_back(std::move(labels));
  }

  return {std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
          std::make_shared<dataset::BoltTokenDataset>(std::move(token_batches)),
          std::make_shared<dataset::BoltDataset>(std::move(label_batches))};
}

BoltGraph getModel() {
  auto input = std::make_shared<Input>(/* dim= */ n_classes);
  auto hidden_layer =
      std::make_shared<FullyConnectedNode>(/* dim= */ 200,
                                           /* activation= */ "relu");
  hidden_layer->addPredecessor(input);

  auto token_input = std::make_shared<TokenInput>();
  auto embedding = std::make_shared<EmbeddingNode>(
      /* num_embedding_lookups= */ 8, /* lookup_size= */ 4,
      /* log_embedding_block_size= */ 12);
  embedding->addInput(token_input);

  auto concat = std::make_shared<ConcatenateNode>();
  concat->setConcatenatedNodes({hidden_layer, embedding});

  auto output =
      std::make_shared<FullyConnectedNode>(/* dim= */ n_classes,
                                           /* activation= */ "softmax");
  output->addPredecessor(concat);

  BoltGraph model(/* inputs= */ {input}, /* token_inputs= */ {token_input},
                  /* output= */ output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  return model;
}

float runDlrmTest(bool noisy_dense_features, bool noisy_categorical_features) {
  auto model = getModel();

  auto [data, tokens, labels] =
      genDlrmDataset(noisy_dense_features, noisy_categorical_features);

  auto train_cfg =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ 2);

  model.train({data}, {tokens}, labels, train_cfg);

  auto predict_cfg =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

  auto [test_metrics, _] = model.predict({data}, {tokens}, labels, predict_cfg);

  std::cout << "ACC: " << test_metrics["categorical_accuracy"] << std::endl;

  return test_metrics["categorical_accuracy"];
}

TEST(DagDlrmTest, SimpleDlrmDataset) {
  float accuracy = runDlrmTest(/*noisy_dense_features= */ false,
                               /* noisy_categorical_features= */ false);
  ASSERT_GE(accuracy, 0.9);
}

TEST(DagDlrmTest, NoisyCategoricalFeatures) {
  float accuracy = runDlrmTest(/*noisy_dense_features= */ false,
                               /* noisy_categorical_features= */ true);
  ASSERT_GE(accuracy, 0.9);
}

TEST(DagDlrmTest, NoisyDenseFeatures) {
  float accuracy = runDlrmTest(/*noisy_dense_features= */ true,
                               /* noisy_categorical_features= */ false);
  ASSERT_GE(accuracy, 0.9);
}

TEST(DagDlrmTest, NoisyDenseAndCategoricalFeatures) {
  float accuracy = runDlrmTest(/*noisy_dense_features= */ true,
                               /* noisy_categorical_features= */ true);
  ASSERT_LE(accuracy, 0.2);
}

}  // namespace thirdai::bolt::tests