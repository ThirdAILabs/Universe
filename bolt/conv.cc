#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/Dataset.h>
#include <chrono>
#include <iostream>
#include <vector>

namespace bolt = thirdai::bolt;
namespace dataset = thirdai::dataset;

int main() {
  std::cout << "Starting Convolution Tests" << std::endl;

  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> train_fac;
  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> test_fac;
  train_fac = std::make_unique<dataset::BoltCsvBatchFactory>(' ');
  test_fac = std::make_unique<dataset::BoltCsvBatchFactory>(' ');

  uint64_t batch_size = 1024;
  dataset::InMemoryDataset<dataset::BoltInputBatch> train_data(
      "/home/david/data/train_birds_4x4.txt", batch_size,
      std::move(*train_fac));
  dataset::InMemoryDataset<dataset::BoltInputBatch> test_data(
      "/home/david/data/test_birds_4x4.txt", batch_size, std::move(*test_fac));

  std::cout << "Finished reading train and test data" << std::endl;

  std::vector<bolt::FullyConnectedLayerConfig> layers;

  uint32_t kernel_size = 4 * 4;
  uint32_t num_patches = 3136;
  // uint32_t kernel_size = 2 * 2;
  // uint32_t num_patches = 196;

  layers.emplace_back(200, 1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 64, 9, 5), kernel_size,
                      num_patches);

  layers.emplace_back(400, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 256, 9, 5), kernel_size, 196);

  layers.emplace_back(800, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 5), 2 * 2, 49);

  layers.emplace_back(20000, .05, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 10));

  layers.emplace_back(325, bolt::ActivationFunction::Softmax);

  bolt::FullyConnectedNetwork network(layers, kernel_size * num_patches);

  auto loss_fn = thirdai::bolt::getLossFunction("categoricalcrossentropyloss");
  std::vector<std::string> metrics = {"categorical_accuracy"};

  float learning_rate = 0.001;
  uint32_t epochs = 25;
  uint32_t rehash = 3000;
  uint32_t rebuild = 10000;
  uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();

  for (uint32_t e = 0; e < epochs; e++) {
    network.train(train_data, *loss_fn, learning_rate, 1, rehash, rebuild,
                  metrics);
    network.predict(test_data, nullptr, metrics, max_test_batches);
  }

  return 0;
}