#pragma once

#include "../layers/FullyConnectedLayer.h"
#include "../utils/DataLoader.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

struct LayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunc act_func;
  SamplingConfig sampling_config;

  static ActivationFunc activationFuncFromStr(const std::string& str) {
    if (str == "ReLU") {
      return ActivationFunc::ReLU;
    }
    if (str == "Softmax") {
      return ActivationFunc::Softmax;
    }
    throw std::invalid_argument(
        "'" + str +
        "' is not a valid activation function. Supported activation "
        "functions: 'ReLU', 'Softmax'");
  }

  static void checkSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }

  LayerConfig(uint64_t _dim, float _sparsity, ActivationFunc _act_func,
              SamplingConfig _config)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config) {}

  LayerConfig(uint64_t _dim, float _sparsity, const std::string& act_func_str,
              SamplingConfig _config)
      : dim(_dim), sparsity(_sparsity), sampling_config(_config) {
    act_func = activationFuncFromStr(act_func_str);
    checkSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, const std::string& act_func_str)
      : dim(_dim), sparsity(1.0), sampling_config(SamplingConfig()) {
    act_func = activationFuncFromStr(act_func_str);
    checkSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, float _sparsity, const std::string& act_func_str)
      : dim(_dim), sparsity(_sparsity) {
    act_func = activationFuncFromStr(act_func_str);
    checkSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t rp = (static_cast<uint32_t>(log2(dim)) / 3) * 3;
      uint32_t k = rp / 3;
      uint32_t rs = (dim * 4) / (1 << rp);
      uint32_t l = sparsity < 0.1 ? 256 : 64;
      sampling_config = SamplingConfig(k, l, rp, rs);
    } else {
      sampling_config = SamplingConfig();
    }
  }

  friend std::ostream& operator<<(std::ostream& out, const LayerConfig& config);
};

class Network {
 public:
  Network(std::vector<LayerConfig> configs, uint64_t input_dim);

  void processTrainingBatch(const Batch& batch, float lr);

  uint32_t processTestBatch(const Batch& batch);

  void train(uint32_t batch_size, const std::string& train_data,
             const std::string& test_data, float learning_rate, uint32_t epochs,
             uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

  uint32_t* predictClasses(const Batch& batch, uint64_t batch_size);

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
      }
    }
    return funcs;
  }

  std::vector<float> getAccuracyPerEpoch() const { return _accuracy_per_epoch; }

  std::vector<int64_t> getTimePerEpoch() const { return _time_per_epoch; }

  float getFinalTestAccuracy() const { return _final_accuracy; }

  ~Network();

 protected:
  std::vector<LayerConfig> _configs;
  uint64_t _input_dim;
  FullyConnectedLayer** _layers;
  uint32_t _num_layers;
  uint64_t _batch_size_hint;
  uint32_t _iter;

  /**
   * Statistics about training accuracy and time per epoch
   */
  std::vector<float> _accuracy_per_epoch;
  std::vector<int64_t> _time_per_epoch;
  float _final_accuracy;
};

}  // namespace thirdai::bolt
