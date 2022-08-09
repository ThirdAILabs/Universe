#pragma once

#include "LayerUtils.h"
#include "SamplingConfig.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {

struct SequentialLayerConfig {
  virtual uint64_t getDim() const = 0;

  virtual float getSparsity() const = 0;

  virtual ActivationFunction getActFunc() const = 0;

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
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using SequentialConfigList =
    std::vector<std::shared_ptr<bolt::SequentialLayerConfig>>;

class FullyConnectedLayerConfig final : public SequentialLayerConfig {
 private:
  uint64_t _dim;
  float _sparsity;
  ActivationFunction _activation_fn;
  SamplingConfigPtr _sampling_config;
  float _adam_compression_factor;

 public:
  // Public constructor - it should only be called by cereal
  FullyConnectedLayerConfig() {}

  FullyConnectedLayerConfig(uint64_t dim, const std::string& activation)
      : FullyConnectedLayerConfig(dim, /* sparsity= */ 1.0, activation,
                                  /*adam_compression_factor=*/1.0) {}

  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            const std::string& activation,
                            float adam_compression_factor = 1.0)
      : FullyConnectedLayerConfig(dim, sparsity, activation,
                                  DWTASamplingConfig::autotune(dim, sparsity),
                                  adam_compression_factor) {}

  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            const std::string& activation,
                            SamplingConfigPtr sampling_config,
                            float adam_compression_factor = 1.0)
      : _dim(dim),
        _sparsity(sparsity),
        _activation_fn(getActivationFunction(activation)),
        _sampling_config(std::move(sampling_config)),
        _adam_compression_factor(adam_compression_factor) {
    if (_sparsity <= 0.0 || _sparsity > 1.0) {
      throw std::invalid_argument(
          "Layer sparsity must be in the range (0.0, 1.0].");
    }

    if (_sparsity < 1.0 && !_sampling_config) {
      throw std::invalid_argument(
          "SamplingConfig cannot be provided as null if sparsity < 1.0.");
    }
  }

  uint64_t getDim() const final { return _dim; }

  float getSparsity() const final { return _sparsity; }

  ActivationFunction getActFunc() const final { return _activation_fn; }

  const SamplingConfigPtr& getSamplingConfig() const {
    return _sampling_config;
  }

  float compression_factor() const { return _adam_compression_factor; }

 private:
  static uint32_t clip(uint32_t input, uint32_t low, uint32_t high) {
    if (input < low) {
      return low;
    }
    if (input > high) {
      return high;
    }
    return input;
  }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<SequentialLayerConfig>(this), _dim, _sparsity,
            _activation_fn, _sampling_config);
  }
};

struct ConvLayerConfig final : public SequentialLayerConfig {
  uint64_t num_filters;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfigPtr sampling_config;
  std::pair<uint32_t, uint32_t> kernel_size;
  uint32_t num_patches;

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func, SamplingConfigPtr _config,
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
      sampling_config = std::make_unique<DWTASamplingConfig>(
          /*num_tables= */ l,
          /* hashes_per_table= */ k, rs);
    } else {
      sampling_config = nullptr;
    }
  }

  uint64_t getDim() const final { return num_filters * num_patches; }

  float getSparsity() const final { return sparsity; }

  ActivationFunction getActFunc() const final { return act_func; }
};

struct EmbeddingLayerConfig {
  uint32_t num_embedding_lookups;
  uint32_t lookup_size;
  uint32_t log_embedding_block_size;

  // Public constructor, needed for cereal to construct optional, should not be
  // used otherwise.
  EmbeddingLayerConfig() {}

  EmbeddingLayerConfig(uint32_t _num_embedding_lookups, uint32_t _lookup_size,
                       uint32_t _log_embedding_block_size)
      : num_embedding_lookups(_num_embedding_lookups),
        lookup_size(_lookup_size),
        log_embedding_block_size(_log_embedding_block_size) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(num_embedding_lookups, lookup_size, log_embedding_block_size);
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedLayerConfig)
