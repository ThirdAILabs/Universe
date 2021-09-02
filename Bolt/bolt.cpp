#include <chrono>
#include <iostream>
#include <vector>

#include "src/Config.h"
#include "src/DataLoader.h"
#include "src/Layer.h"
#include "src/Network.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid args, usage: ./slide <config file>" << std::endl;
    return 1;
  }

  ConfigReader config(argv[1]);
  config.PrintConfigVals();

  uint32_t num_layers = config.IntVal("num_layers");
  std::vector<bolt::LayerConfig> layers;

  for (uint32_t l = 0; l < num_layers; l++) {
    bolt::ActivationFunc func = bolt::ActivationFunc::ReLU;
    if (l == num_layers - 1) {
      func = bolt::ActivationFunc::Softmax;
    }
    layers.push_back(
        {config.IntVal("dims", l), config.FloatVal("sparsity", l), func,
         bolt::SamplingConfig(config.IntVal("K", l), config.IntVal("L", l),
                              config.IntVal("range_pow", l), config.IntVal("reservoir_size", l))});
  }

  bolt::Network network(layers, config.IntVal("input_dim"));

  float learning_rate = config.FloatVal("learning_rate");
  uint32_t epochs = config.IntVal("epochs");
  uint32_t batch_size = config.IntVal("batch_size");
  uint32_t rehash = config.IntVal("rehash");
  uint32_t rebuild = config.IntVal("rebuild");

  uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();
  if (config.ValExists("max_test_batches")) {
    max_test_batches = config.IntVal("max_test_batches");
  }

  network.Train(batch_size, config.StrVal("train_data"), config.StrVal("test_data"), learning_rate,
                epochs, rehash, rebuild, max_test_batches);

  return 0;
}