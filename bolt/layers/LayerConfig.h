#pragma once

#include <cmath>
#include <fstream>
#include <iostream>

namespace thirdai::bolt {

enum class ActivationFunc { ReLU, Softmax, MeanSquared };

struct SamplingConfig {
  uint32_t hashes_per_table, num_tables, range_pow, reservoir_size;

  SamplingConfig()
      : hashes_per_table(0), num_tables(0), range_pow(0), reservoir_size(0) {}

  SamplingConfig(uint32_t hashes_per_table, uint32_t num_tables,
                 uint32_t range_pow, uint32_t reservoir_size)
      : hashes_per_table(hashes_per_table),
        num_tables(num_tables),
        range_pow(range_pow),
        reservoir_size(reservoir_size) {}
};

struct FullyConnectedLayerConfig {
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
    if (str == "MeanSquared") {
      return ActivationFunc::MeanSquared;
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

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunc _act_func, SamplingConfig _config)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config) {}

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            const std::string& act_func_str,
                            SamplingConfig _config)
      : dim(_dim), sparsity(_sparsity), sampling_config(_config) {
    act_func = activationFuncFromStr(act_func_str);
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            const std::string& act_func_str)
      : dim(_dim), sparsity(_sparsity) {
    act_func = activationFuncFromStr(act_func_str);
    checkSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t k = static_cast<uint32_t>(ceil(log2(dim) / 4));
      uint32_t rp = k * 3;
      uint32_t rs = (dim * 4) / (1 << rp);
      uint32_t l = sparsity < 0.05 ? 64 : 32;
      sampling_config = SamplingConfig(k, l, rp, rs);
    } else {
      sampling_config = SamplingConfig();
    }
  }

  FullyConnectedLayerConfig(uint64_t _dim, const std::string& act_func_str, 
                            bool autotune=false)
      : dim(_dim) {
    act_func = activationFuncFromStr(act_func_str);
    if (autotune) {
      uint32_t k = static_cast<uint32_t>(ceil(log2(dim) / 4));
      uint32_t rp = k * 3;
      uint32_t rs = (dim * 4) / (1 << rp);
      sparsity = static_cast<float>(pow(0.35, k));
      uint32_t l = sparsity < 0.05 ? 64 : 32;
      sampling_config = SamplingConfig(k, l, rp, rs); 
    } else {
      sparsity = 1.0;
      sampling_config = SamplingConfig();
    }
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  const FullyConnectedLayerConfig& config) {
    out << "Layer: dim=" << config.dim << ", load_factor=" << config.sparsity;
    switch (config.act_func) {
      case ActivationFunc::ReLU:
        out << ", act_func=ReLU";
        break;
      case ActivationFunc::Softmax:
        out << ", act_func=Softmax";
        break;
      case ActivationFunc::MeanSquared:
        out << ", act_func=MeanSquared";
        break;
    }
    if (config.sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << config.sampling_config.hashes_per_table
          << ", num_tables=" << config.sampling_config.num_tables
          << ", range_pow=" << config.sampling_config.range_pow
          << ", reservoir_size=" << config.sampling_config.reservoir_size
          << "}";
    }
    return out;
  }
};

struct EmbeddingLayerConfig {
  uint32_t num_embedding_lookups;
  uint32_t lookup_size;
  uint32_t log_embedding_block_size;

  EmbeddingLayerConfig(uint32_t _num_embedding_lookups, uint32_t _lookup_size,
                       uint32_t _log_embedding_block_size)
      : num_embedding_lookups(_num_embedding_lookups),
        lookup_size(_lookup_size),
        log_embedding_block_size(_log_embedding_block_size) {}
};

}  // namespace thirdai::bolt