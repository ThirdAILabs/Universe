#include "TestDatasetGenerators.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <gtest/gtest.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt::tests {

static constexpr uint32_t N_CLASSES = 100, N_BATCHES = 100, BATCH_SIZE = 100;

BoltGraph getTrainedModel() {
  auto input = Input::make(N_CLASSES);

  auto output = FullyConnectedNode::makeDense(N_CLASSES, "softmax")
                    ->addPredecessor(input);

  BoltGraph model({input}, output);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [train_data, train_labels] =
      TestDatasetGenerators::generateSimpleVectorDataset(
          N_CLASSES, N_BATCHES, BATCH_SIZE, /* noisy_dataset= */ false);

  TrainConfig train_config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ 2)
          .silence();

  model.train({train_data}, train_labels, train_config);

  return model;
}

double testModel(BoltGraph& model, dataset::BoltDatasetPtr& test_data,
                 dataset::BoltDatasetPtr& test_labels) {
  PredictConfig predict_config = PredictConfig::makeConfig()
                                     .withMetrics({"categorical_accuracy"})
                                     .silence();

  auto [metrics, _] = model.predict({test_data}, test_labels, predict_config);

  return metrics["categorical_accuracy"];
}

TEST(PredictSingleTest, PredictSingle) {
  BoltGraph model = getTrainedModel();

  auto [test_data, test_labels] =
      TestDatasetGenerators::generateSimpleVectorDataset(
          N_CLASSES, N_BATCHES, BATCH_SIZE, /* noisy_dataset= */ false);

  double predict_acc = testModel(model, test_data, test_labels);

  uint32_t correct = 0, total = 0;

  for (uint32_t batch_idx = 0; batch_idx < test_data->numBatches();
       batch_idx++) {
    for (uint32_t vec_idx = 0; vec_idx < test_data->batchSize(batch_idx);
         vec_idx++) {
      auto output = model.predictSingle({test_data->at(batch_idx)[vec_idx]},
                                        /* use_sparse_inference= */ false);

      uint32_t prediction = output.getHighestActivationId();
      uint32_t label = test_labels->at(batch_idx)[vec_idx].active_neurons[0];

      if (prediction == label) {
        correct++;
      }
      total++;
    }
  }

  double predict_single_acc = static_cast<double>(correct) / total;

  ASSERT_GT(predict_single_acc, 0.90);
  ASSERT_DOUBLE_EQ(predict_single_acc, predict_acc);
}

TEST(PredictSingleTest, PredictBatch) {
  BoltGraph model = getTrainedModel();

  auto [test_data, test_labels] =
      TestDatasetGenerators::generateSimpleVectorDataset(
          N_CLASSES, N_BATCHES, BATCH_SIZE, /* noisy_dataset= */ false);

  double predict_acc = testModel(model, test_data, test_labels);

  uint32_t correct = 0, total = 0;

  for (uint32_t batch_idx = 0; batch_idx < test_data->numBatches();
       batch_idx++) {
    std::vector<BoltBatch> inputs;
    inputs.push_back(std::move(test_data->at(batch_idx)));

    BoltBatch outputs = model.predictSingleBatch(
        std::move(inputs), /* use_sparse_inference= */ false);

    for (uint32_t vec_idx = 0; vec_idx < test_labels->batchSize(batch_idx);
         vec_idx++) {
      uint32_t prediction = outputs[vec_idx].getHighestActivationId();
      uint32_t label = test_labels->at(batch_idx)[vec_idx].active_neurons[0];

      if (prediction == label) {
        correct++;
      }
      total++;
    }
  }

  std::cout << correct << " " << total << std::endl;

  double predict_single_acc = static_cast<double>(correct) / total;

  ASSERT_GT(predict_single_acc, 0.90);
  ASSERT_DOUBLE_EQ(predict_single_acc, predict_acc);
}

}  // namespace thirdai::bolt::tests
