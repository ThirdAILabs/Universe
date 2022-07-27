#include "BoltNetworkTestUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

namespace thirdai::bolt::tests {

static constexpr uint32_t n_classes = 100;

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);

  network.train(data, labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.98);
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainNoisyDatasetSingleLayerNetwork) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ true);

  network.train(data, labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 5,
                /* rehash= */ 0, /* rebuild= */ 0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics = network.predict(
      data, labels, /* output_active_neurons= */ nullptr,
      /* output_activations= */ nullptr, /* use_sparse_inference= */ false,
      /* metric_names= */ {"categorical_accuracy"},
      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

static void testSimpleDatasetMultiLayerNetworkActivation(
    const std::string& act) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(10000, 0.1, act),
       std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);

  auto train_metrics =
      network.train(data, labels, CategoricalCrossEntropyLoss(),
                    /* learning_rate */ 0.001, /* epochs */ 2,
                    /* rehash= */ 0, /* rebuild= */ 0,
                    /* metric_names= */ {"mean_squared_error"},
                    /* verbose= */ false);
  ASSERT_LT(train_metrics.at("mean_squared_error").back(),
            train_metrics.at("mean_squared_error").front());

  auto test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetMultiLayerNetwork) {
  testSimpleDatasetMultiLayerNetworkActivation("relu");
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetMultiLayerNetworkTanh) {
  testSimpleDatasetMultiLayerNetworkActivation("tanh");
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetMultiLayerNetworkSigmoid) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(10000, 0.1, "relu"),
       std::make_shared<FullyConnectedLayerConfig>(n_classes, "sigmoid")},
      n_classes);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);

  auto train_metrics =
      network.train(data, labels, CategoricalCrossEntropyLoss(),
                    /* learning_rate= */ 0.001, /* epochs= */ 5,
                    /* rehash= */ 0, /* rebuild= */ 0,
                    /* metric_names= */ {"mean_squared_error"},
                    /* verbose= */ true);

  ASSERT_LT(train_metrics.at("mean_squared_error").back(),
            train_metrics.at("mean_squared_error").front());

  auto test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ true);
  ASSERT_GE(test_metrics["categorical_accuracy"], 0.99);
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainNoisyDatasetMultiLayerNetwork) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(10000, 0.1, "relu"),
       std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  auto [data, labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ true);

  network.train(data, labels, CategoricalCrossEntropyLoss(),
                /* learning_rate= */ 0.001, /* epochs= */ 2,
                /* rehash= */ 0, /* rebuild=*/0, /* metric_names= */ {},
                /* verbose= */ false);
  auto test_metrics =
      network.predict(data, labels, /* output_active_neurons= */ nullptr,
                      /* output_activations= */ nullptr,
                      /* use_sparse_inference= */ false,
                      /* metric_names= */ {"categorical_accuracy"},
                      /* verbose= */ false);
  ASSERT_LE(test_metrics["categorical_accuracy"], 0.2);
}

TEST(FullyConnectedClassificationNetworkTest, MultiLayerNetworkToString) {
  FullyConnectedNetwork network(
      {/* layer1= */ std::make_shared<FullyConnectedLayerConfig>(
           /* dim= */ 10000, /* sparsity= */ 0.1,
           /* act_func= */ "relu"),
       /* layer2= */ std::make_shared<FullyConnectedLayerConfig>(
           /* dim= */ n_classes, /* act_func= */ "softmax")},
      /* input_dim= */ n_classes);

  std::stringstream summary;
  network.buildNetworkSummary(summary);

  std::string expected =
      "========= Bolt Network =========\n"
      "InputLayer (Layer 0): dim=100\n"
      "FullyConnectedLayer (Layer 1): dim=10000, sparsity=0.1, act_func=ReLU\n"
      "FullyConnectedLayer (Layer 2): dim=100, sparsity=1, act_func=Softmax\n"
      "================================";
  std::string actual = summary.str();

  std::cout << actual << std::endl;

  ASSERT_EQ(expected, actual);
}

// This doesn't need to do anything, just needs to implement the DataLoader
// interface so that we can construct a mock streaming dataset. See comment
// below for more details on how this test works.
class DummyDataLoader final : public dataset::DataLoader {
 public:
  explicit DummyDataLoader(uint32_t max_batches)
      : DataLoader(/* target_batch_size = */ 100),
        _batch_counter(0),
        _max_batches(max_batches) {}

  std::optional<std::vector<std::string>> nextBatch() final {
    if (_batch_counter >= _max_batches) {
      return std::nullopt;
    }
    _batch_counter++;
    return {{}};
  }

  std::optional<std::string> getHeader() final { return ""; }

  std::string resourceName() const final { return ""; }

 private:
  uint32_t _batch_counter, _max_batches;
};

/*
  Mock batch processor that consumes an InMemoryDataset and returns its
  batches. The idea behind the batch processor is that it will receive raw rows
  from the dataset and convert them into the given batch type to be processed
  by bolt. In these tests we are not interested in testing the data
  loader/batch processer functionality as this is handled seperately, and soley
  interested in testing that bolt trains correctly on streaming datasets. Thus
  we create a mock batch processor that instead of processing actual rows and
  returning batches, just returns already created batches in order from an in
  memory dataset. Having the functionality to return a batch when nextBatch() is
  called is sufficient to construct a streaming dataset, in addition to the
  DummDataLoader defined above.
*/
class MockBatchProcessor final
    : public dataset::BatchProcessor<BoltBatch, BoltBatch> {
 public:
  MockBatchProcessor(dataset::BoltDatasetPtr data,
                     dataset::BoltDatasetPtr labels)
      : _data(std::move(data)), _labels(std::move(labels)), _batch_counter(0) {}

  std::tuple<BoltBatch, BoltBatch> createBatch(
      const std::vector<std::string>& rows) final {
    (void)rows;

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

std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
getMockStreamingDataset(dataset::BoltDatasetPtr& data,
                        dataset::BoltDatasetPtr& labels) {
  std::shared_ptr<dataset::DataLoader> mock_loader =
      std::make_shared<DummyDataLoader>(data->numBatches());

  std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>
      mock_processor = std::make_shared<MockBatchProcessor>(data, labels);

  return std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
      mock_loader, mock_processor);
}

void testFullyConnectedNetworkOnStream(FullyConnectedNetwork& network,
                                       uint32_t epochs, float acc_threshold) {
  for (uint32_t e = 0; e < epochs; e++) {
    auto [in_mem_data, in_mem_labels] =
        genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);
    auto stream_data = getMockStreamingDataset(in_mem_data, in_mem_labels);

    network.trainOnStream(stream_data, CategoricalCrossEntropyLoss(),
                          /* learning_rate= */ 0.001,
                          /* rehash_batch= */ 10, /* rebuild_batch= */ 50,
                          /* metric_names= */ {},
                          /* metric_log_batch_interval=*/0,
                          /* verbose= */ false);
  }

  auto [in_mem_data, in_mem_labels] =
      genDataset(/* n_classes= */ n_classes, /* noisy_dataset= */ false);
  auto stream_data = getMockStreamingDataset(in_mem_data, in_mem_labels);

  auto test_metrics =
      network.predictOnStream(stream_data, /* use_sparse_inference= */ false,
                              /* metric_names= */ {"categorical_accuracy"},
                              /* batch_callback= */ std::nullopt,
                              /* verbose= */ false);
  ASSERT_GE(test_metrics["categorical_accuracy"], acc_threshold);
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetSingleLayerNetworkStreamingData) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  testFullyConnectedNetworkOnStream(network, /* epochs= */ 5,
                                    /* acc_threshold= */ 0.98);
}

TEST(FullyConnectedClassificationNetworkTest,
     TrainSimpleDatasetMultiLayerNetworkStreamingData) {
  FullyConnectedNetwork network(
      {std::make_shared<FullyConnectedLayerConfig>(
           /* dim=*/10000, /* sparsity= */ 0.1, "relu"),
       std::make_shared<FullyConnectedLayerConfig>(n_classes, "softmax")},
      n_classes);

  testFullyConnectedNetworkOnStream(network, /* epochs= */ 2,
                                    /* acc_threshold= */ 0.99);
}

}  // namespace thirdai::bolt::tests