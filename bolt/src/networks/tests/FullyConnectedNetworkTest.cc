#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <algorithm>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

static const uint32_t n_classes = 100, n_batches = 100, batch_size = 100;

class FullyConnectedClassificationNetworkTestFixture : public testing::Test {
 public:
  static dataset::DatasetWithLabels genDataset(bool add_noise) {
    std::mt19937 gen(892734);
    std::uniform_int_distribution<uint32_t> label_dist(0, n_classes - 1);
    std::normal_distribution<float> data_dist(0, add_noise ? 1.0 : 0.1);

    std::vector<bolt::BoltBatch> data_batches;
    std::vector<bolt::BoltBatch> label_batches;
    for (uint32_t b = 0; b < n_batches; b++) {
      std::vector<bolt::BoltVector> labels;
      std::vector<bolt::BoltVector> vectors;
      for (uint32_t i = 0; i < batch_size; i++) {
        uint32_t label = label_dist(gen);
        bolt::BoltVector v(n_classes, true, false);
        std::generate(v.activations, v.activations + n_classes,
                      [&]() { return data_dist(gen); });
        if (!add_noise) {
          v.activations[label] += 1.0;
        }
        vectors.push_back(std::move(v));
        labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
      }
      data_batches.push_back(bolt::BoltBatch(std::move(vectors)));
      label_batches.push_back(bolt::BoltBatch(std::move(labels)));
    }

    return dataset::DatasetWithLabels(
        dataset::BoltDataset(std::move(data_batches), n_batches * batch_size),
        dataset::BoltDataset(std::move(label_batches), n_batches * batch_size));
  }
};

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.98);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainNoisyDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

static void testSimpleDatasetMultiLayerNetworkActivation(
    ActivationFunction act) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(10000, 0.1, act),
       std::make_shared<FullyConnectedLayerConfig>(
           n_classes, ActivationFunction::Softmax)},
      n_classes);

  auto data = FullyConnectedClassificationNetworkTestFixture::genDataset(false);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate */ 0.001, /* epochs */ 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetwork) {
  testSimpleDatasetMultiLayerNetworkActivation(ActivationFunction::ReLU);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetworkTanh) {
  testSimpleDatasetMultiLayerNetworkActivation(ActivationFunction::Tanh);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetworkSigmoid) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Sigmoid)},
                                n_classes);

  auto data = FullyConnectedClassificationNetworkTestFixture::genDataset(false);

  network.train(data.data, data.labels, BinaryCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 2,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ true);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ true);
  // Lower accuracy threshold to 0.6 because Sigmoid/BCE converges slower than
  // ReLU/Tanh.
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.6);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainNoisyDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Softmax)},
                                n_classes);

  auto data = genDataset(true);

  network.train(data.data, data.labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 2,
                /* rehash= */ 0, /* rebuild=*/0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data.data, data.labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

class DummyDataLoader final : public dataset::DataLoader {
 public:
  DummyDataLoader() : DataLoader(batch_size) {}

  std::optional<std::vector<std::string>> nextBatch() final { return {{}}; }

  std::optional<std::string> getHeader() final { return ""; }

  std::string resourceName() const final { return ""; }
};

// Mock batch processor that consumes an InMemoryDataset and returns its
// batches.
class MockBatchProcessor final : public dataset::BatchProcessor<BoltBatch> {
 public:
  MockBatchProcessor(dataset::BoltDatasetPtr data,
                     dataset::BoltDatasetPtr labels)
      : _data(std::move(data)), _labels(std::move(labels)), _batch_counter(0) {}

  std::optional<dataset::BoltDataLabelPair<BoltBatch>> createBatch(
      const std::vector<std::string>& rows) final {
    (void)rows;

    if (_batch_counter >= _data->numBatches()) {
      return std::nullopt;
    }

    std::pair<BoltBatch, BoltBatch> batch_pair = {
        std::move(_data->at(_batch_counter)),
        std::move(_labels->at(_batch_counter))};
    _batch_counter++;

    return batch_pair;
  }

  bool expectsHeader() const final { return false; }

  void processHeader(const std::string& header) final { (void)header; }

 private:
  dataset::BoltDatasetPtr _data;
  dataset::BoltDatasetPtr _labels;
  uint32_t _batch_counter;
};

std::shared_ptr<dataset::StreamingDataset<BoltBatch>> getMockStreamingDataset(
    dataset::DatasetWithLabels&& dataset) {
  std::shared_ptr<dataset::DataLoader> mock_loader =
      std::make_shared<DummyDataLoader>();

  std::shared_ptr<dataset::BatchProcessor<BoltBatch>> mock_processor =
      std::make_shared<MockBatchProcessor>(dataset.data, dataset.labels);

  return std::make_shared<dataset::StreamingDataset<BoltBatch>>(mock_loader,
                                                                mock_processor);
}

void testFullyConnectedNetworkOnStream(FullyConnectedNetwork& network,
                                       uint32_t epochs, float acc_threshold) {
  for (uint32_t e = 0; e < epochs; e++) {
    auto in_mem_data =
        FullyConnectedClassificationNetworkTestFixture::genDataset(false);
    auto stream_data = getMockStreamingDataset(std::move(in_mem_data));

    network.trainOnStream(stream_data, CategoricalCrossEntropyLoss(),
                          /* learning_rate= */ 0.001,
                          /* rehash_batch= */ 10, /* rebuild_batch= */ 50,
                          /* metric_names= */ {},
                          /* metric_log_batch_interval=*/0,
                          /* verbose= */ false);
  }

  auto in_mem_data =
      FullyConnectedClassificationNetworkTestFixture::genDataset(false);
  auto stream_data = getMockStreamingDataset(std::move(in_mem_data));

  auto test_metrics =
      network.predictOnStream(stream_data,
                              /* metric_names= */ {"categorical_accuracy"},
                              /* batch_callback= */ std::nullopt,
                              /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], acc_threshold);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetSingleLayerNetworkStreamingData) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                    n_classes, ActivationFunction::Softmax)},
                                n_classes);

  testFullyConnectedNetworkOnStream(network, 5, 0.98);
}

TEST_F(FullyConnectedClassificationNetworkTestFixture,
       TrainSimpleDatasetMultiLayerNetworkStreamingData) {
  FullyConnectedNetwork network({std::make_shared<FullyConnectedLayerConfig>(
                                     10000, 0.1, ActivationFunction::ReLU),
                                 std::make_shared<FullyConnectedLayerConfig>(
                                     n_classes, ActivationFunction::Softmax)},
                                n_classes);

  testFullyConnectedNetworkOnStream(network, 2, 0.99);
}

}  // namespace thirdai::bolt::tests