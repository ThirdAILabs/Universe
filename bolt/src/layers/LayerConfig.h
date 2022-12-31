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
      : FullyConnectedLayerConfig(dim, sparsity, activation,
                                  DWTASamplingConfig::autotune(dim, sparsity)) {
  }

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

class ConvLayerConfig final {
 public:
  ConvLayerConfig(uint64_t _num_filters, const std::string& _activation,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  std::pair<uint32_t, uint32_t> _next_kernel_size)
      : ConvLayerConfig(_num_filters, /* sparsity= */ 1.0, _activation,
                        _kernel_size, _next_kernel_size) {}

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  const std::string& _activation,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  std::pair<uint32_t, uint32_t> _next_kernel_size)
      : ConvLayerConfig(_num_filters, _sparsity, _activation, _kernel_size,
                        _next_kernel_size,
                        DWTASamplingConfig::autotune(_num_filters, _sparsity)) {
  }

  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  const std::string& _activation,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  std::pair<uint32_t, uint32_t> _next_kernel_size,
                  SamplingConfigPtr _config);

  uint64_t num_filters;
  float sparsity;
  ActivationFunction activation_fn;
  std::pair<uint32_t, uint32_t> kernel_size;
  std::pair<uint32_t, uint32_t> next_kernel_size;
  SamplingConfigPtr sampling_config;

 private:
  // Private constructor for cereal
  ConvLayerConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(num_filters, sparsity, activation_fn, sampling_config, kernel_size);
  }
};

enum class EmbeddingReductionType {
  SUM,
  CONCATENATION,
};

class EmbeddingLayerConfig {
 public:
  // Public constructor, needed for cereal to construct optional, should not be
  // used otherwise.
  EmbeddingLayerConfig() {}

  EmbeddingLayerConfig(
      uint32_t num_embedding_lookups, uint32_t lookup_size,
      uint32_t log_embedding_block_size, const std::string& reduction,
      std::optional<uint32_t> num_tokens_per_input = std::nullopt);

  uint32_t numEmbeddingLookups() const { return _num_embedding_lookups; }

  uint32_t lookupSize() const { return _lookup_size; }

  uint32_t logEmbeddingBlockSize() const { return _log_embedding_block_size; }

  EmbeddingReductionType reduction() const { return _reduction; }

  uint32_t getOutputDim() const {
    uint32_t output_dim = _num_embedding_lookups * _lookup_size;
    if (_reduction == EmbeddingReductionType::CONCATENATION) {
      output_dim *= _num_tokens_per_input.value();
    }
    return output_dim;
  }

  std::optional<uint32_t> numTokensPerInput() const {
    return _num_tokens_per_input;
  }

 private:
  uint32_t _num_embedding_lookups;
  uint32_t _lookup_size;
  uint32_t _log_embedding_block_size;

  EmbeddingReductionType _reduction;
  std::optional<uint32_t> _num_tokens_per_input;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  static EmbeddingReductionType getReductionType(
      const std::string& reduction_name);
};

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
