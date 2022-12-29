#include "LayerConfig.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>
#include <utility>

namespace thirdai::bolt {

void checkSparsity(float sparsity) {
  if (sparsity > 1 || sparsity <= 0) {
    throw std::invalid_argument(
        "sparsity must be between 0 exclusive and 1 inclusive.");
  }
  if (0.2 < sparsity && sparsity < 1.0) {
    std::cout << "WARNING: Using large sparsity value " << sparsity
              << " in Layer, consider decreasing sparsity" << std::endl;
  }
}

template <class Archive>
void FullyConnectedLayerConfig::serialize(Archive& archive) {
  archive(_dim, _sparsity, _activation_fn, _sampling_config);
}

template <class Archive>
void EmbeddingLayerConfig::serialize(Archive& archive) {
  archive(_num_embedding_lookups, _lookup_size, _log_embedding_block_size,
          _reduction, _num_tokens_per_input);
}

FullyConnectedLayerConfig::FullyConnectedLayerConfig(
    uint64_t dim, float sparsity, const std::string& activation,
    SamplingConfigPtr sampling_config)
    : _dim(dim),
      _sparsity(sparsity),
      _activation_fn(getActivationFunction(activation)),
      _sampling_config(std::move(sampling_config)) {
  if (_sparsity <= 0.0 || _sparsity > 1.0) {
    throw std::invalid_argument(
        "Layer sparsity must be in the range (0.0, 1.0].");
  }

  if (_sparsity < 1.0 && !_sampling_config) {
    throw std::invalid_argument(
        "SamplingConfig cannot be provided as null if sparsity < 1.0.");
  }
}

ConvLayerConfig::ConvLayerConfig(
    uint64_t _num_filters, float _sparsity, const std::string& _activation,
    std::pair<uint32_t, uint32_t> _kernel_size, uint32_t _num_patches,
    std::pair<uint32_t, uint32_t> _next_kernel_size, SamplingConfigPtr _config)
    : num_filters(_num_filters),
      sparsity(_sparsity),
      activation_fn(getActivationFunction(_activation)),
      kernel_size(std::move(_kernel_size)),
      num_patches(_num_patches),
      next_kernel_size(std::move(_next_kernel_size)),
      sampling_config(std::move(_config)) {
  checkSparsity(sparsity);
}

EmbeddingLayerConfig::EmbeddingLayerConfig(
    uint32_t num_embedding_lookups, uint32_t lookup_size,
    uint32_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint32_t> num_tokens_per_input)
    : _num_embedding_lookups(num_embedding_lookups),
      _lookup_size(lookup_size),
      _log_embedding_block_size(log_embedding_block_size),
      _reduction(getReductionType(reduction)),
      _num_tokens_per_input(num_tokens_per_input) {
  if (_reduction == EmbeddingReductionType::CONCATENATION &&
      !_num_tokens_per_input) {
    throw std::invalid_argument(
        "Cannot construct embedding layer with concatenation reduction "
        "without specifying num_tokens_per_input.");
  }
}

EmbeddingReductionType EmbeddingLayerConfig::getReductionType(
    const std::string& reduction_name) {
  std::string lower_name = utils::lower(reduction_name);
  if (lower_name == "sum") {
    return EmbeddingReductionType::SUM;
  }
  if (lower_name == "concat" || lower_name == "concatenation") {
    return EmbeddingReductionType::CONCATENATION;
  }
  throw std::invalid_argument(
      "Invalid embedding reduction time '" + reduction_name +
      "', supported options are 'sum' or 'concat'/'concatenation'");
}

template <class Archive>
void NormalizationLayerConfig::serialize(Archive& archive) {
  archive(_beta_regularizer, _gamma_regularizer, _epsilon);
}

template void EmbeddingLayerConfig::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);
template void EmbeddingLayerConfig::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void FullyConnectedLayerConfig::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);

template void FullyConnectedLayerConfig::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void NormalizationLayerConfig::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);
template void NormalizationLayerConfig::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

}  // namespace thirdai::bolt
