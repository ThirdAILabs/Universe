#include "networks/Network.h"
#include "utils/ConfigReader.h"
#include <dataset/src/Dataset.h>
#include <chrono>
#include <iostream>
#include <vector>

namespace bolt = thirdai::bolt;
namespace dataset = thirdai::dataset;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid args, usage: ./bolt <config file>" << std::endl;
    return 1;
  }

  bolt::ConfigReader config(argv[1]);

  uint32_t num_layers = config.intVal("num_layers");
  std::vector<bolt::FullyConnectedLayerConfig> layers;

  for (uint32_t l = 0; l < num_layers; l++) {
    bolt::ActivationFunc func = bolt::ActivationFunc::ReLU;
    if (l == num_layers - 1) {
      func = bolt::ActivationFunc::Softmax;
    }
    layers.emplace_back(
        config.intVal("dims", l), config.floatVal("sparsity", l), func,
        bolt::SamplingConfig(config.intVal("hashes_per_table", l),
                             config.intVal("num_tables", l),
                             config.intVal("range_pow", l),
                             config.intVal("reservoir_size", l)));
  }

  bolt::Network network(layers, config.intVal("input_dim"));

  float learning_rate = config.floatVal("learning_rate");
  uint32_t epochs = config.intVal("epochs");
  uint32_t batch_size = config.intVal("batch_size");
  uint32_t rehash = config.intVal("rehash");
  uint32_t rebuild = config.intVal("rebuild");
  uint32_t switch_inference_epoch = config.valExists("sparse_inf_at_epoch")
                                        ? config.intVal("sparse_inf_at_epoch")
                                        : 0;

  uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();
  if (config.valExists("max_test_batches")) {
    max_test_batches = config.intVal("max_test_batches");
  }

  if (!config.valExists("dataset_format")) {
    std::cerr << "dataset_format is not specified in the config file."
              << std::endl;
    return 1;
  }
  try {
    auto train_data =
        dataset::InMemoryDataset<dataset::BoltInputBatch>::loadBoltDataset(
            config.strVal("train_data"), batch_size,
            config.strVal("dataset_format"));

    auto test_data =
        dataset::InMemoryDataset<dataset::BoltInputBatch>::loadBoltDataset(
            config.strVal("test_data"), batch_size,
            config.strVal("dataset_format"));

    for (uint32_t e = 0; e < epochs; e++) {
      if ((switch_inference_epoch > 0) && e == switch_inference_epoch) {
        std::cout << "Freezing Selection for Sparse Inference" << std::endl;
        network.useSparseInference();
      }

      network.train(train_data, learning_rate, 1, rehash, rebuild);
      network.predict(test_data, max_test_batches);
    }
    network.predict(test_data);
  } catch (std::exception& e) {
    std::cerr << "Error reading dataset" << std::endl;
  }

  return 0;
}
