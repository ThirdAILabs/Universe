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

    uint64_t batch_size = 64;
    dataset::InMemoryDataset<dataset::BoltInputBatch> train_data(
        "/home/david/data/train_mnist2x2.txt", batch_size,
        std::move(*train_fac));
    dataset::InMemoryDataset<dataset::BoltInputBatch> test_data(
        "/home/david/data/test_mnist2x2.txt", batch_size,
        std::move(*test_fac));

    std::cout << "Finished reading train and test data" << std::endl;

    std::vector<bolt::FullyConnectedLayerConfig> layers;

    uint32_t patch_size = 4;
    uint32_t num_patches = 196;

    layers.emplace_back(100, 1, bolt::ActivationFunction::ReLU, bolt::SamplingConfig(4, 256, 12, 10), patch_size, num_patches);

    layers.emplace_back(200, 1, bolt::ActivationFunction::ReLU, bolt::SamplingConfig(4, 256, 12, 10), 2*2*100, 49);

    layers.emplace_back(800, 0.1, bolt::ActivationFunction::ReLU, bolt::SamplingConfig(4, 256, 12, 10));

    layers.emplace_back(10, bolt::ActivationFunction::Softmax);

    bolt::FullyConnectedNetwork network(layers, patch_size * num_patches);

    auto loss_fn =
      thirdai::bolt::getLossFunction("categoricalcrossentropyloss");
    std::vector<std::string> metrics = {"categorical_accuracy"};

    float learning_rate = 0.0001;
    uint32_t epochs = 25;
    uint32_t rehash = 1000;
    uint32_t rebuild = 1000; // 0.2 times number of images times patches per image
    uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();      

    for (uint32_t e = 0; e < epochs; e++) {
      network.train(train_data, *loss_fn, learning_rate, 1, rehash, rebuild, metrics);
      network.predict(test_data, nullptr, metrics, max_test_batches);
    }

    return 0;
}