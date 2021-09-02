#pragma once

#include <math.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <limits>

#include "DataLoader.h"
#include "Layer.h"

#ifdef PYTHON_BINDINGS
#include <pybind11/numpy.h>

namespace py = pybind11;
#endif

namespace bolt {

struct LayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunc act_func;
  SamplingConfig sampling_config;

  static ActivationFunc ActivationFuncFromStr(const std::string& str) {
    if (str == "ReLU") {
      return ActivationFunc::ReLU;
    } else if (str == "Softmax") {
      return ActivationFunc::Softmax;
    } else {
      throw std::invalid_argument("'" + str +
                                  "' is not a valid activation function. Supported activation "
                                  "functions: 'ReLU', 'Softmax'");
    }
  }

  static void CheckSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }

  LayerConfig(uint64_t _dim, float _sparsity, ActivationFunc _act_func, SamplingConfig _config)
      : dim(_dim), sparsity(_sparsity), act_func(_act_func), sampling_config(_config) {}

  LayerConfig(uint64_t _dim, float _sparsity, std::string act_func_str, SamplingConfig _config)
      : dim(_dim), sparsity(_sparsity), sampling_config(_config) {
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, std::string act_func_str)
      : dim(_dim), sparsity(1.0), sampling_config(SamplingConfig()) {
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
  }

  LayerConfig(uint64_t _dim, float _sparsity, std::string act_func_str)
      : dim(_dim), sparsity(_sparsity) {
    act_func = ActivationFuncFromStr(act_func_str);
    CheckSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t rp = (log2(dim) / 3) * 3;
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

  void Train(uint32_t batch_size, std::string train_data, std::string test_data,
             float learning_rate, uint32_t epochs, uint32_t rehash = 0, uint32_t rebuild = 0,
             uint32_t max_test_batches = std::numeric_limits<uint32_t>::max());

#ifdef PYTHON_BINDINGS

  py::array_t<uint32_t> PredictClasses(const Batch& batch, uint64_t batch_size);

  // TODO: this duplicates the weight vector to ensure safety, alteratives:
  //        1. Take the weight matrix and set pointers to null
  //        2. Allow for duplicate reference (unsafe)
  py::array_t<float> GetWeightMatrix(uint32_t layer_index);

  py::array_t<float> GetBiasVector(uint32_t layer_index);

#endif

  void ReBuildHashFunctions();

  void BuildHashTables();

  ~Network();

 private:
  std::vector<LayerConfig> configs;
  uint64_t input_dim;
  Layer** layers;
  uint32_t num_layers;
  uint64_t batch_size_hint;
  uint32_t iter;
};

}  // namespace bolt
