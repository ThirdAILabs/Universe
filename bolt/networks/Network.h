#pragma once

#include "../layers/FullyConnectedLayer.h"
#include "../utils/DataLoader.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Network {
 public:
  Network(std::vector<FullyConnectedLayerConfig> configs, uint64_t input_dim);

  void processTrainingBatch(const Batch& batch, float lr);

  uint32_t processTestBatch(const Batch& batch);

  void train(uint32_t batch_size, const std::string& train_data,
             const std::string& test_data, float learning_rate, uint32_t epochs,
             uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

  void reBuildHashFunctions();

  void buildHashTables();

  uint32_t getNumLayers() const { return _num_layers; }

  uint32_t getInputDim() const { return _input_dim; }

  std::vector<uint32_t> getLayerSizes() {
    std::vector<uint32_t> layer_sizes;
    for (const auto& c : _configs) {
      layer_sizes.push_back(c.dim);
    }
    return layer_sizes;
  }

  std::vector<std::string> getActivationFunctions() {
    std::vector<std::string> funcs;
    for (const auto& c : _configs) {
      switch (c.act_func) {
        case ActivationFunc::ReLU:
          funcs.emplace_back("ReLU");
          break;

        case ActivationFunc::Softmax:
          funcs.emplace_back("Softmax");
          break;

        case ActivationFunc::MeanSquared:
          funcs.emplace_back("MeanSquared");
          break;
      }
    }
    return funcs;
  }

  std::vector<float> getAccuracyPerEpoch() const { return _accuracy_per_epoch; }

  std::vector<int64_t> getTimePerEpoch() const { return _time_per_epoch; }

  float getFinalTestAccuracy() const { return _final_accuracy; }

  ~Network();

 protected:
  std::vector<FullyConnectedLayerConfig> _configs;
  uint64_t _input_dim;
  FullyConnectedLayer** _layers;
  uint32_t _num_layers;
  uint32_t _iter;

  /**
   * Statistics about training accuracy and time per epoch
   */
  std::vector<float> _accuracy_per_epoch;
  std::vector<int64_t> _time_per_epoch;
  float _final_accuracy;
};

}  // namespace thirdai::bolt