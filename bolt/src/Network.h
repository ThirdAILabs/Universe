#pragma once

#include "DataLoader.h"
#include "Layer.h"
#include <iostream>
#include <limits>
#include <math.h>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

struct LayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunc act_func;
  SamplingConfig sampling_config;

  static ActivationFunc ActivationFuncFromStr(const std::string& str) {
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

  static void CheckSparsity(float sparsity) {
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
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, const std::string& act_func_str)
      : dim(_dim), sparsity(1.0), sampling_config(SamplingConfig()) {
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, float _sparsity, const std::string& act_func_str)
      : dim(_dim), sparsity(_sparsity) {
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
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
};

class Network {
 public:
  Network(std::vector<LayerConfig> configs, uint64_t input_dim);

  void ProcessTrainingBatch(const Batch& batch, float lr);

  uint32_t ProcessTestBatch(const Batch& batch);

  void Train(uint32_t batch_size, const std::string& train_data,
             const std::string& test_data, float learning_rate, uint32_t epochs,
             uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

  uint32_t* PredictClasses(const Batch& batch, uint64_t batch_size);

  void ReBuildHashFunctions();

  void BuildHashTables();

  uint32_t GetNumLayers() const { return num_layers; }

  uint32_t GetInputDim() const { return input_dim; }

  std::vector<uint32_t> GetLayerSizes() {
    std::vector<uint32_t> layer_sizes;
    for (const auto& c : configs) {
      layer_sizes.push_back(c.dim);
    }
    return layer_sizes;
  }

  std::vector<std::string> GetActivationFunctions() {
    std::vector<std::string> funcs;
    for (const auto& c : configs) {
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

  ~Network();

 protected:
  std::vector<LayerConfig> configs;
  uint64_t input_dim;
  Layer** layers;
  uint32_t num_layers;
  uint64_t batch_size_hint;
  uint32_t iter;
};

}  // namespace thirdai::bolt
