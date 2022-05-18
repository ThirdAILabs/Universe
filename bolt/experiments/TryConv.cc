#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/Dataset.h>
#include <chrono>
#include <iostream>
#include <vector>

namespace bolt = thirdai::bolt;
namespace dataset = thirdai::dataset;

int main() {  // NOLINT exceptions
  std::cout << "Starting Convolution Tests" << std::endl;

  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> factory;
  factory =
      std::make_unique<dataset::BoltCsvBatchFactory>(/* delimiter = */ ' ');

  uint64_t batch_size = 1024;
  dataset::InMemoryDataset<dataset::BoltInputBatch> train_data(
      "/Users/david/Documents/python_/train_mnist2x2.txt", batch_size,
      std::move(*factory));
  dataset::InMemoryDataset<dataset::BoltInputBatch> test_data(
      "/Users/david/Documents/python_/test_mnist2x2.txt", batch_size,
      std::move(*factory));

  std::cout << "Finished reading train and test data" << std::endl;

  std::pair<uint32_t, uint32_t> kernel_size(2, 2);
  uint32_t num_patches = 196;

  bolt::SequentialConfigList layers;

  layers.push_back(std::make_shared<bolt::ConvLayerConfig>(
      16, 1, bolt::ActivationFunction::ReLU, bolt::SamplingConfig(3, 64, 9, 5),
      kernel_size, num_patches));

  layers.push_back(std::make_shared<bolt::ConvLayerConfig>(
      200, .1, bolt::ActivationFunction::ReLU,
      bolt::SamplingConfig(3, 256, 9, 5), kernel_size, 49));

  layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
      1000, .1, bolt::ActivationFunction::ReLU,
      bolt::SamplingConfig(4, 256, 12, 10)));

  layers.push_back(std::make_shared<bolt::FullyConnectedLayerConfig>(
      10, bolt::ActivationFunction::Softmax));

  bolt::FullyConnectedNetwork network(
      layers, kernel_size.first * kernel_size.second * num_patches, {});

  auto loss_fn = thirdai::bolt::CategoricalCrossEntropyLoss();
  std::vector<std::string> metrics = {"categorical_accuracy"};

  float learning_rate = 0.001;
  uint32_t epochs = 25;
  uint32_t rehash = 3000;
  uint32_t rebuild = 10000;
  uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();

  for (uint32_t e = 0; e < epochs; e++) {
    network.train(train_data, loss_fn, learning_rate, 1, rehash, rebuild,
                  metrics);
    network.predict(test_data, nullptr, nullptr, metrics, max_test_batches);
  }

  return 0;
}