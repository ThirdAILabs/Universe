#include "LayerConfig.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>

namespace thirdai::bolt {

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

ConvLayerConfig::ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                                 ActivationFunction _act_func,
                                 SamplingConfigPtr _config,
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

ConvLayerConfig::ConvLayerConfig(uint64_t _num_filters, float _sparsity,
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
    uint32_t hashes_per_table = rp / 3;
    uint32_t reservoir_size = (num_filters * 4) / (1 << rp);
    uint32_t num_tables = sparsity < 0.1 ? 256 : 64;
    sampling_config = std::make_unique<DWTASamplingConfig>(
        /* num_tables= */ num_tables,
        /* hashes_per_table= */ hashes_per_table, /* range_pow= */ 3,
        /* binsize=*/8, reservoir_size, /*permutations=*/4);
  } else {
    sampling_config = nullptr;
  }
}

EmbeddingLayerConfig::EmbeddingLayerConfig(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input)
    : EmbeddingLayerConfig(
          /* num_embedding_lookups= */ num_embedding_lookups,
          /* lookup_size= */ lookup_size,
          /* log_embedding_block_size= */ log_embedding_block_size,
          /* update_chunk_size= */ DEFAULT_EMBEDDING_UPDATE_CHUNK_SIZE,
          /* reduction= */ reduction,
          /* num_tokens_per_input= */ num_tokens_per_input) {}

EmbeddingLayerConfig::EmbeddingLayerConfig(
    uint64_t num_embedding_lookups, uint64_t lookup_size,
    uint64_t log_embedding_block_size, uint64_t update_chunk_size,
    const std::string& reduction, std::optional<uint64_t> num_tokens_per_input)
    : _num_embedding_lookups(num_embedding_lookups),
      _lookup_size(lookup_size),
      _log_embedding_block_size(log_embedding_block_size),
      _update_chunk_size(update_chunk_size),
      _reduction(reductionFromString(reduction)),
      _num_tokens_per_input(num_tokens_per_input) {
  if (_reduction == EmbeddingReductionType::CONCATENATION &&
      !_num_tokens_per_input) {
    throw std::invalid_argument(
        "Cannot construct embedding layer with concatenation reduction "
        "without specifying num_tokens_per_input.");
  }
}

EmbeddingReductionType reductionFromString(const std::string& name) {
  std::string lower_name = text::lower(name);
  if (lower_name == "sum") {
    return EmbeddingReductionType::SUM;
  }
  if (lower_name == "concat" || lower_name == "concatenation") {
    return EmbeddingReductionType::CONCATENATION;
  }
  if (lower_name == "average" || lower_name == "avg") {
    return EmbeddingReductionType::AVERAGE;
  }
  throw std::invalid_argument("Invalid embedding reduction time '" + name +
                              "', supported options are 'sum', "
                              "'average'/'avg', or 'concat'/'concatenation'");
}

std::string reductionToString(EmbeddingReductionType reduction) {
  switch (reduction) {
    case EmbeddingReductionType::SUM:
      return "sum";
    case EmbeddingReductionType::CONCATENATION:
      return "concat";
    case EmbeddingReductionType::AVERAGE:
      return "avg";
    default:
      return "";
  }
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
