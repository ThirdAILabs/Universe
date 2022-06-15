#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <algorithm>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

static const uint32_t n_classes = 100, n_batches = 100, batch_size = 100;

class DLRMTestFixture : public testing::Test {
 public:
  static dataset::ClickThroughDatasetWithLabels genDataset(
      bool noisy_dense, bool noisy_categorical) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, noisy_dense ? 1.0 : 0.1);

    std::vector<dataset::ClickThroughBatch> data_batches;
    std::vector<bolt::BoltBatch> label_batches;
    for (uint32_t b = 0; b < n_batches; b++) {
      std::vector<bolt::BoltVector> labels;
      std::vector<bolt::BoltVector> dense_features;
      std::vector<std::vector<uint32_t>> categorical_features;
      for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t label = label_dist(gen);
        bolt::BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!noisy_dense) {
          v.activations[label] += 1.0;
        }
        dense_features.push_back(std::move(v));
        categorical_features.push_back(
            {noisy_categorical ? label + label_dist(gen) : label});
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.push_back(dataset::ClickThroughBatch(
          std::move(dense_features), std::move(categorical_features)));
      label_batches.push_back(bolt::BoltBatch(std::move(labels)));
    }

    return dataset::ClickThroughDatasetWithLabels(
        dataset::InMemoryDataset<dataset::ClickThroughBatch>(
            std::move(data_batches), n_batches * batch_size),
        dataset::BoltDataset(std::move(label_batches), n_batches * batch_size));
  }
};

// TODO(Vihan): Figure out why this gets such low accuracy, but not with -O0
// TEST_F(DLRMTestFixture, SimpleDataset) {
//   std::vector<FullyConnectedLayerConfig> bottom_mlp = {
//       FullyConnectedLayerConfig(200, ActivationFunction::ReLU)};

//   EmbeddingLayerConfig embedding_layer = EmbeddingLayerConfig(8, 16, 12);

//   std::vector<FullyConnectedLayerConfig> top_mlp = {
//       FullyConnectedLayerConfig(1000, 0.1, ActivationFunction::ReLU,
//                                 SamplingConfig(2, 32, 6, 32)),
//       FullyConnectedLayerConfig(n_classes, ActivationFunction::Softmax)};

//   DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, n_classes);

//   auto dataset = genDataset(false, false);

//   dlrm.train(dataset.data, dataset.labels, CategoricalCrossEntropyLoss(),
//   0.001,
//              6);
//   auto test_metrics = dlrm.predict(dataset.data, dataset.labels, nullptr,
//                                    nullptr, {"categorical_accuracy"});

//   ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
// }

TEST_F(DLRMTestFixture, NoisyCategoricalFeatures) {
  SequentialConfigList bottom_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(200,
                                                  ActivationFunction::ReLU)};

  EmbeddingLayerConfig embedding_layer = EmbeddingLayerConfig(8, 16, 12);

  SequentialConfigList top_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(
          1000, 0.1, ActivationFunction::ReLU, SamplingConfig(2, 32, 6, 32)),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, n_classes);

  auto dataset = genDataset(false, true);

  dlrm.train(dataset.data, dataset.labels, CategoricalCrossEntropyLoss(), 0.001,
             32, /* rehash= */ 0,
             /* rebuild= */ 0, /* metric_names= */ {},
             /* verbose= */ false);
  auto test_metrics = dlrm.predict(dataset.data, dataset.labels,
                                   /* output_active_neurons= */ nullptr,
                                   /* output_activations= */ nullptr,
                                   /* metric_names= */ {"categorical_accuracy"},
                                   /* verbose= */ false);

  ASSERT_GE(test_metrics["categorical_accuracy"], 0.9);
}

TEST_F(DLRMTestFixture, NoisyDenseFeatures) {
  SequentialConfigList bottom_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(200,
                                                  ActivationFunction::ReLU)};

  EmbeddingLayerConfig embedding_layer = EmbeddingLayerConfig(8, 16, 12);

  SequentialConfigList top_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(
          1000, 0.1, ActivationFunction::ReLU, SamplingConfig(2, 32, 6, 32)),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, n_classes);

  auto dataset = genDataset(true, false);

  dlrm.train(dataset.data, dataset.labels, CategoricalCrossEntropyLoss(), 0.001,
             3, /* rehash= */ 0,
             /* rebuild= */ 0, /* metric_names= */ {},
             /* verbose= */ false);
  auto test_metrics = dlrm.predict(dataset.data, dataset.labels,
                                   /* output_active_neurons= */ nullptr,
                                   /* output_activations= */ nullptr,
                                   /* metric_names= */ {"categorical_accuracy"},
                                   /* verbose= */ false);

  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST_F(DLRMTestFixture, NoisyDenseAndCategoricalFeatures) {
  SequentialConfigList bottom_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(200,
                                                  ActivationFunction::ReLU)};

  EmbeddingLayerConfig embedding_layer = EmbeddingLayerConfig(8, 16, 12);

  SequentialConfigList top_mlp = {
      std::make_shared<FullyConnectedLayerConfig>(
          1000, 0.1, ActivationFunction::ReLU, SamplingConfig(2, 32, 6, 32)),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, n_classes);

  auto dataset = genDataset(true, true);

  dlrm.train(dataset.data, dataset.labels, CategoricalCrossEntropyLoss(), 0.001,
             5, /* rehash= */ 0,
             /* rebuild= */ 0, /* metric_names= */ {},
             /* verbose= */ false);
  auto test_metrics = dlrm.predict(dataset.data, dataset.labels,
                                   /* output_active_neurons= */ nullptr,
                                   /* output_activations= */ nullptr,
                                   /* metric_names= */ {"categorical_accuracy"},
                                   /* verbose= */ false);

  ASSERT_LE(test_metrics["categorical_accuracy"], 0.1);
}

}  // namespace thirdai::bolt::tests