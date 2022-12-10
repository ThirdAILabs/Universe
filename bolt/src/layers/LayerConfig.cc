#include "LayerConfig.h"

namespace thirdai::bolt {

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

ConvLayerConfig::ConvLayerConfig(uint64_t num_filters, float sparsity,
                                 const std::string& activation,
                                 std::pair<uint32_t, uint32_t> kernel_size,
                                 SamplingConfigPtr config)
    : _num_filters(num_filters),
      _sparsity(sparsity),
      _activation_fn(getActivationFunction(activation)),
      _kernel_size(std::move(kernel_size)),
      _sampling_config(std::move(config)) {
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

}  // namespace thirdai::bolt