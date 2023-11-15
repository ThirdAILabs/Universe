#pragma once

#include "LayerUtils.h"
#include "SamplingConfig.h"
#include <utils/StringManipulation.h>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace thirdai::bolt {

class FullyConnectedLayerConfig {
 private:
  uint64_t _dim;
  float _sparsity;
  ActivationFunction _activation_fn;
  SamplingConfigPtr _sampling_config;

 public:
  // Public constructor - it should only be called by cereal
  FullyConnectedLayerConfig() {}

  FullyConnectedLayerConfig(uint64_t dim, const std::string& activation)
      : FullyConnectedLayerConfig(dim, /* sparsity= */ 1.0, activation) {}

  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            const std::string& activation)
      : FullyConnectedLayerConfig(
            dim, sparsity, activation,
            DWTASamplingConfig::autotune(dim, sparsity,
                                         /* experimental_autotune=*/false)) {}

  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            const std::string& activation,
                            SamplingConfigPtr sampling_config);

  uint64_t getDim() const { return _dim; }

  float getSparsity() const { return _sparsity; }

  ActivationFunction getActFunc() const { return _activation_fn; }

  const SamplingConfigPtr& getSamplingConfig() const {
    return _sampling_config;
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

struct ConvLayerConfig {
  uint64_t num_filters;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfigPtr sampling_config;
  std::pair<uint32_t, uint32_t> kernel_size;
  uint32_t num_patches;

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func, SamplingConfigPtr _config,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  uint32_t _num_patches);

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  uint32_t _num_patches);

  uint64_t getDim() const { return num_filters * num_patches; }

  float getSparsity() const { return sparsity; }

  ActivationFunction getActFunc() const { return act_func; }
};

enum class EmbeddingReductionType {
  SUM,
  CONCATENATION,
  AVERAGE,
};

/**
 * During sparse updates in the embedding table the only chunks of the embedding
 * table which are accessed are updated. This parameter controls the size of
 * those chunks. The default value 4 was determined by doing a search over
 * different values on the criteo dataset using different batch sizes, embedding
 * table sizes, and lookup sizes.
 */
constexpr uint32_t DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE = 4;

class EmbeddingLayerConfig {
 public:
  // Public constructor, needed for cereal to construct optional, should not be
  // used otherwise.
  EmbeddingLayerConfig() {}

  EmbeddingLayerConfig(
      uint64_t num_embedding_lookups, uint64_t lookup_size,
      uint64_t log_embedding_block_size, const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input = std::nullopt);

  EmbeddingLayerConfig(
      uint64_t num_embedding_lookups, uint64_t lookup_size,
      uint64_t log_embedding_block_size, uint64_t update_chunk_size,
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input = std::nullopt);

  uint64_t numEmbeddingLookups() const { return _num_embedding_lookups; }

  uint64_t lookupSize() const { return _lookup_size; }

  uint64_t logEmbeddingBlockSize() const { return _log_embedding_block_size; }

  uint64_t updateChunkSize() const { return _update_chunk_size; }

  EmbeddingReductionType reduction() const { return _reduction; }

  uint64_t getOutputDim() const {
    uint64_t output_dim = _num_embedding_lookups * _lookup_size;
    if (_reduction == EmbeddingReductionType::CONCATENATION) {
      output_dim *= _num_tokens_per_input.value();
    }
    return output_dim;
  }

  std::optional<uint64_t> numTokensPerInput() const {
    return _num_tokens_per_input;
  }

 private:
  uint64_t _num_embedding_lookups;
  uint64_t _lookup_size;
  uint64_t _log_embedding_block_size;
  uint64_t _update_chunk_size;

  EmbeddingReductionType _reduction;
  std::optional<uint64_t> _num_tokens_per_input;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

EmbeddingReductionType reductionFromString(const std::string& name);

std::string reductionToString(EmbeddingReductionType reduction);

class NormalizationLayerConfig {
 public:
  explicit NormalizationLayerConfig(float epsilon = 0.00001)
      : _epsilon(epsilon),
        _beta_regularizer(std::nullopt),
        _gamma_regularizer(std::nullopt) {}

  static NormalizationLayerConfig makeConfig() {
    return NormalizationLayerConfig();
  }

  NormalizationLayerConfig& setCenteringFactor(float centering_factor) {
    _beta_regularizer = centering_factor;
    return *this;
  }

  NormalizationLayerConfig& setScalingFactor(float scaling_factor) {
    _gamma_regularizer = scaling_factor;
    return *this;
  }

  // The default value for _beta_regularizer is 0.0 (has no effect)
  constexpr float beta() const {
    return _beta_regularizer.has_value() ? _beta_regularizer.value() : 0.0;
  }

  // The default value for _gamma_regularizer is 1.0 (has no effect)
  constexpr float gamma() const {
    return _gamma_regularizer.has_value() ? _gamma_regularizer.value() : 1.0;
  }
  constexpr float epsilon() const { return _epsilon; }

 private:
  // small threshold added to avoid division by zero
  float _epsilon;

  // If beta_regularizer and gamma_regularizer are set, then the z-score
  // will be scaled (multiplied) by a factor of _gamma_regularizer and
  // shifted (centered) by a factor of _beta_regularizer
  std::optional<float> _beta_regularizer;
  std::optional<float> _gamma_regularizer;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::bolt
