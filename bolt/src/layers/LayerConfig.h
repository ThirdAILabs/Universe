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
    if (sparsity > 1 || sparsity <= 0) {
      throw std::invalid_argument(
          "sparsity must be between 0 exclusive and 1 inclusive.");
    }
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

    // We don't need a sampling config for a layer with sparsity equal to 1.0,
    // so we can just return (the default value of the sampling_config will
    // be everything set to 0s)
    if (sparsity == 1.0) {
      return;
    }

    // The number of items in the table is equal to the number of neurons in 
    // this layer, which is stored in the "dim" variable. By analyzing the
    // hash table, we find that
    // E(num_elements_per_bucket) = dim / 2^(range_pow) = sparsity * dim * safety_factor / num_tables
    // The first expression comes from analyzing a single hash table, while 
    // the second comes from analyzing the total number of elements returned
    // across the tables. safety_factor is a constant that equals how many more 
    // times elements we want to expect to have across tables than the minimum.
    // Simplifying, we have
    // 1 / 2^(range_pow) = sparsity * C / num_tables
    // This leaves us with 3 free variables: C, num_tables, and hashes_per_table. 
    
    // First, we will set num_tables = 128 and C = 2. This is just a heuristic.
    uint32_t num_tables = 128;
    uint32_t safety_factor = 2;

    // We can now set range_pow: manipulating the equation, we have that
    // range_pow = log2(num_tables / (sparsity * safety_factor))
    uint32_t range_pow = std::log2(num_tables / (sparsity * safety_factor));
    // By the properties of DWTA, hashes_per_table = range_pow / 3. 
    uint32_t hashes_per_table = range_pow / 3;
    // Since range_pow might originally not have been a direct multiple of 3,
    // we reset it here to be equal to hashes_per_table * 3. This might lower
    // range_pow and mean that there are more elements per bucket than the 
    // equation calls for, but that is better than raising it and there being 
    // too few.
    range_pow = hashes_per_table * 3;


    // Finally, we want to set reservoir_size to be somewhat larger than
    // the number of expected elements per bucket. Here, we choose as a heuristic
    // 2 times the number of expected elements per bucket.
    uint32_t expected_num_elements_per_bucket = dim / (1 << range_pow);
    uint32_t reservoir_size = 2 * expected_num_elements_per_bucket;

    sampling_config = SamplingConfig(/* hashes_per_table = */ hashes_per_table,
                                     /* num_tables = */ num_tables,
                                     /* range_pow = */ range_pow,
                                     /* reservoir_size = */ reservoir_size);
  }

  uint64_t getDim() const { return dim; }

  float getSparsity() const { return sparsity; }

  ActivationFunction getActFunc() const { return act_func; }

 private:
  template <typename T>
  static T clip(T input, T low, T high) {
    if (input < low) {
      return low;
    }
    if (input > high) {
      return high;
    }
    return input;
  }

  void print(std::ostream& out) const {
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

  uint64_t getDim() const { return num_filters * num_patches; }

  float getSparsity() const { return sparsity; }

  ActivationFunction getActFunc() const { return act_func; }

 private:
  void print(std::ostream& out) const {
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