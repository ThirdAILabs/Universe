#pragma once

#include <cereal/types/vector.hpp>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {

enum class ActivationFunction { ReLU, Softmax, Linear };

static ActivationFunction getActivationFunction(
    const std::string& act_func_name) {
  std::string lower_name;
  for (char c : act_func_name) {
    lower_name.push_back(std::tolower(c));
  }
  if (lower_name == "relu") {
    return ActivationFunction::ReLU;
  }
  if (lower_name == "softmax") {
    return ActivationFunction::Softmax;
  }
  if (lower_name == "linear") {
    return ActivationFunction::Linear;
  }
  throw std::invalid_argument("'" + act_func_name +
                              "' is not a valid activation function");
}

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

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(hashes_per_table, num_tables, range_pow, reservoir_size);
  }
};

struct FullyConnectedLayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;
  uint32_t patch_dim = 0;
  uint32_t num_patches = 0;

  static void checkSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func,
                            SamplingConfig _config,
                            uint64_t _patch_dim = 0, uint64_t _num_patches = 0)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config),
        patch_dim(_patch_dim),
        num_patches(_num_patches) {
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t _dim, ActivationFunction _act_func)
      : dim(_dim),
        sparsity(1.0),
        act_func(_act_func),
        sampling_config(SamplingConfig()) {
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func)
      : dim(_dim), sparsity(_sparsity), act_func(_act_func) {
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

  friend std::ostream& operator<<(std::ostream& out,
                                  const FullyConnectedLayerConfig& config) {
    out << "Layer: dim=" << config.dim << ", load_factor=" << config.sparsity;
    switch (config.act_func) {
      case ActivationFunction::ReLU:
        out << ", act_func=ReLU";
        break;
      case ActivationFunction::Softmax:
        out << ", act_func=Softmax";
        break;
      case ActivationFunction::Linear:
        out << ", act_func=Linear";
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
    if (config.patch_dim != 0) {
      out << ", patch_dim=" << config.patch_dim
          << ", num_patches=" << config.num_patches;
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