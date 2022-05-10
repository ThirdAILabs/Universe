#pragma once

#include "LayerUtils.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace thirdai::bolt {

struct SequentialLayerConfig {
  virtual uint64_t getDim() const = 0;

  virtual float getSparsity() const = 0;

  virtual ActivationFunction getActFunc() const = 0;

  friend std::ostream& operator<<(std::ostream& out,
                                  const SequentialLayerConfig& config) {
    config.print(out);
    return out;
  }

  static void checkSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }

 private:
  virtual void print(std::ostream& out) const = 0;
};

using SequentialConfigList =
    std::vector<std::shared_ptr<bolt::SequentialLayerConfig>>;

struct FullyConnectedLayerConfig final : public SequentialLayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func,
                            SamplingConfig _config)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config) {
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

  uint64_t getDim() const final { return dim; }

  float getSparsity() const final { return sparsity; }

  ActivationFunction getActFunc() const final { return act_func; }

 private:
  void print(std::ostream& out) const final {
    out << "FullyConnected: dim=" << dim << ", load_factor=" << sparsity;
    switch (act_func) {
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
    if (sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << sampling_config.hashes_per_table
          << ", num_tables=" << sampling_config.num_tables
          << ", range_pow=" << sampling_config.range_pow
          << ", reservoir_size=" << sampling_config.reservoir_size << "}";
    }
  }
};

struct ConvLayerConfig final : public SequentialLayerConfig {
  uint64_t num_filters;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;
  std::pair<uint32_t, uint32_t> kernel_size;
  uint32_t num_patches;

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func, SamplingConfig _config,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  uint32_t _num_patches)
      : num_filters(_num_filters),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config),
        kernel_size(std::move(_kernel_size)),
        num_patches(_num_patches) {
    checkSparsity(sparsity);
  }

  uint64_t getDim() const final { return num_filters * num_patches; }

  float getSparsity() const final { return sparsity; }

  ActivationFunction getActFunc() const final { return act_func; }

 private:
  void print(std::ostream& out) const final {
    out << "Conv: num_filters=" << num_filters << ", load_factor=" << sparsity
        << ", num_patches=" << num_patches;
    switch (act_func) {
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
    out << ", kernel_size: (" << kernel_size.first << ", " << kernel_size.second
        << ")";
    if (sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << sampling_config.hashes_per_table
          << ", num_tables=" << sampling_config.num_tables
          << ", range_pow=" << sampling_config.range_pow
          << ", reservoir_size=" << sampling_config.reservoir_size << "}";
    }
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