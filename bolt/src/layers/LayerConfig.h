#pragma once

#include "LayerUtils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>

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
      std::cout << "WARNING: Using large sparsity value " << sparsity
                << " in Layer, consider decreasing sparsity" << std::endl;
    }
  }

 private:
  virtual void print(std::ostream& out) const = 0;
};

using SequentialConfigList =
    std::vector<std::shared_ptr<bolt::SequentialLayerConfig>>;
// edited
inline SamplingConfig initialize_hash_function(hashing_function type_of_hash,
                                               uint64_t _dim, float _sparsity) {
  uint32_t num_tables_guess = 128;
  float safety_factor = 1;
  float range_pow_float =
      std::log2(num_tables_guess / (_sparsity * safety_factor));
  float hashes_per_table_float = range_pow_float / 3;
  uint32_t hashes_per_table = std::round(hashes_per_table_float);
  hashes_per_table = clip(hashes_per_table, /* low = */ 2, /* high = */ 8);
  uint32_t range_pow = hashes_per_table * 3;
  uint32_t num_tables =
      std::round(_sparsity * safety_factor * (1 << range_pow));
  uint32_t expected_num_elements_per_bucket =
      std::max<uint32_t>(_dim / (1 << range_pow), 1);
  uint32_t reservoir_size = 4 * expected_num_elements_per_bucket;
  std::string ht;
  if (type_of_hash == hashing_function::Densified_MinHash) {
    ht = "DensifiedMinHash";
  }
  if (type_of_hash == hashing_function::DWTA) {
    ht = "DWTA";
  }
  if (type_of_hash == hashing_function::SRP) {
    ht = "SRP";
  }
  if (type_of_hash == hashing_function::Fast_SRP) {
    ht = "FastSRP";
  }
  return {/* hashes_per_table = */ hashes_per_table,
          /* num_tables = */ num_tables,
          /* range_pow = */ range_pow,
          /* reservoir_size = */ reservoir_size,
          /*default hash_type*/ ht};  // edited
}
// edited

struct FullyConnectedLayerConfig final : public SequentialLayerConfig {
  uint64_t dim;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            const std::string& _act_func,
                            SamplingConfig _config)
      : FullyConnectedLayerConfig(_dim, _sparsity,
                                  getActivationFunction(_act_func), _config) {}

  FullyConnectedLayerConfig(uint64_t _dim, const std::string& _act_func)
      : FullyConnectedLayerConfig(_dim, getActivationFunction(_act_func)) {}

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            const std::string& _act_func)
      : FullyConnectedLayerConfig(_dim, _sparsity,
                                  getActivationFunction(_act_func)) {}

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func,
                            SamplingConfig _config)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(std::move(_config)) {  // edited
    checkSparsity(sparsity);
    // edited
    if (_sparsity == 1.0) {
      return;
    }
    if (sampling_config.hashes_per_table != 0 ||
        sampling_config.num_tables != 0 || sampling_config.range_pow != 0 ||
        sampling_config.reservoir_size != 0) {
      return;
    }
    hashing_function hash_name = get_hash_function(sampling_config.hash_type);
    sampling_config = initialize_hash_function(hash_name, dim, sparsity);
    // edited
  }

  FullyConnectedLayerConfig(uint64_t _dim, ActivationFunction _act_func)
      : FullyConnectedLayerConfig(_dim, 1.0, _act_func) {}
  //     : dim(_dim),
  //       sparsity(1.0),
  //       act_func(_act_func),
  //       sampling_config(SamplingConfig()) {
  //   checkSparsity(sparsity);
  // }

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func)
      : FullyConnectedLayerConfig(_dim, _sparsity, _act_func,
                                  SamplingConfig()) {}
  //   : dim(_dim), sparsity(_sparsity), act_func(_act_func),
  //   sampling_config(initialize_hash_function(hashing_function::DWTAHashFunction,_dim,_sparsity))
  //   {
  // checkSparsity(sparsity);
  // edited
  // // We don't need a sampling config for a layer with sparsity equal to 1.0,
  // // so we can just return (the default value of the sampling_config will
  // // be everything set to 0s)
  // if (sparsity == 1.0) {
  //   return;
  // }

  // // The number of items in the table is equal to the number of neurons in
  // // this layer, which is stored in the "dim" variable. By analyzing the
  // // hash table, we find that
  // // E(num_elements_per_bucket) = dim / 2^(range_pow) = sparsity * dim *
  // // safety_factor / num_tables The first expression comes from analyzing a
  // // single hash table, while the second comes from analyzing the total
  // number
  // // of elements returned across the tables. safety_factor is a constant that
  // // equals how many more times elements we want to expect to have across
  // // tables than the minimum. Simplifying, we have 1 / 2^(range_pow) =
  // // sparsity * safety_factor / num_tables This leaves us with 3 free
  // // variables: safety_factor, num_tables, and hashes_per_table.

  // // First, we will set num_tables_guess = 128 and safety_factor = 1.
  // // num_tables_guess is an initial guess to get a good value for range_pow,
  // // but we do not find the final num_tables until below because the rounding
  // // in the range_pow calculation step can mess things up.
  // uint32_t num_tables_guess = 128;
  // float safety_factor = 1;

  // // We can now set range_pow: manipulating the equation, we have that
  // // range_pow = log2(num_tables / (sparsity * safety_factor))
  // float range_pow_float =
  //     std::log2(num_tables_guess / (sparsity * safety_factor));
  // // By the properties of DWTA, hashes_per_table = range_pow / 3.
  // float hashes_per_table_float = range_pow_float / 3;
  // // We now round hashes_per_table to the nearest integer.
  // // Using round is more accurate than truncating it down.
  // uint32_t hashes_per_table = std::round(hashes_per_table_float);
  // // Finally, hashes_per_table needs to be clipped, and then we can
  // // recalculate range_pow
  // hashes_per_table = clip(hashes_per_table, /* low = */ 2, /* high = */ 8);
  // uint32_t range_pow = hashes_per_table * 3;

  // // We now calculate an exact value for num_tables using the formula
  // // num_tables = sparsity * safety_factor * 2^(range_pow)
  // uint32_t num_tables =
  //     std::round(sparsity * safety_factor * (1 << range_pow));

  // // Finally, we want to set reservoir_size to be somewhat larger than
  // // the number of expected elements per bucket. Here, we choose as a
  // // heuristic 4 times the number of expected elements per bucket. We take
  // // a max with 1 to ensure that the reservoir size isn't 0.
  // uint32_t expected_num_elements_per_bucket =
  //     std::max<uint32_t>(dim / (1 << range_pow), 1);
  // uint32_t reservoir_size = 4 * expected_num_elements_per_bucket;

  // sampling_config = SamplingConfig(/* hashes_per_table = */ hashes_per_table,
  //                                  /* num_tables = */ num_tables,
  //                                  /* range_pow = */ range_pow,
  //                                  /* reservoir_size = */ reservoir_size,
  //                                  /*default hash_type*/"DWTA");//edited

  uint64_t getDim() const final { return dim; }

  float getSparsity() const final { return sparsity; }

  ActivationFunction getActFunc() const final { return act_func; }

 private:
  void print(std::ostream& out) const final {
    out << "FullyConnected: dim=" << dim << ", sparsity=" << sparsity;
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
      case ActivationFunction::Tanh:
        out << ", act_func=Tanh";
        break;
    }
    if (sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << sampling_config.hashes_per_table
          << ", num_tables=" << sampling_config.num_tables
          << ", range_pow=" << sampling_config.range_pow
          << ", reservoir_size=" << sampling_config.reservoir_size
          << ", hash_type=" << sampling_config.hash_type << "}";  // edited
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
        sampling_config(std::move(_config)),
        kernel_size(std::move(_kernel_size)),
        num_patches(_num_patches) {
    checkSparsity(sparsity);
  }

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  uint32_t _num_patches)
      : num_filters(_num_filters),
        sparsity(_sparsity),
        act_func(_act_func),
        kernel_size(std::move(_kernel_size)),
        num_patches(_num_patches) {
    checkSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t rp = (static_cast<uint32_t>(log2(num_filters)) / 3) * 3;
      uint32_t k = rp / 3;
      uint32_t rs = (num_filters * 4) / (1 << rp);
      uint32_t l = sparsity < 0.1 ? 256 : 64;
      sampling_config = SamplingConfig(k, l, rp, rs, "DWTA");  // edited
    } else {
      sampling_config = SamplingConfig();
    }
  }

  uint64_t getDim() const final { return num_filters * num_patches; }

  float getSparsity() const final { return sparsity; }

  ActivationFunction getActFunc() const final { return act_func; }

 private:
  void print(std::ostream& out) const final {
    out << "Conv: num_filters=" << num_filters << ", sparsity=" << sparsity
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
      case ActivationFunction::Tanh:
        out << ", act_func=Tanh";
        break;
    }
    out << ", kernel_size: (" << kernel_size.first << ", " << kernel_size.second
        << ")";
    if (sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << sampling_config.hashes_per_table
          << ", num_tables=" << sampling_config.num_tables
          << ", range_pow=" << sampling_config.range_pow
          << ", reservoir_size=" << sampling_config.reservoir_size
          << ", hash_type=" << sampling_config.hash_type << "}";  // edited
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