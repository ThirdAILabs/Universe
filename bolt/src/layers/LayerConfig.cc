#include "LayerConfig.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/optional.hpp>

namespace thirdai::bolt {

template <class Archive>
void SequentialLayerConfig::serialize(Archive& archive) {
  (void)archive;
}

void SequentialLayerConfig::checkSparsity(float sparsity) {
  if (sparsity > 1 || sparsity <= 0) {
    throw std::invalid_argument(
        "sparsity must be between 0 exclusive and 1 inclusive.");
  }
  if (0.2 < sparsity && sparsity < 1.0) {
    std::cout << "WARNING: Using large sparsity value " << sparsity
              << " in Layer, consider decreasing sparsity" << std::endl;
  }
}

uint32_t FullyConnectedLayerConfig::clip(uint32_t input, uint32_t low,
                                         uint32_t high) {
  if (input < low) {
    return low;
  }
  if (input > high) {
    return high;
  }
  return input;
}

template <class Archive>
void FullyConnectedLayerConfig::serialize(Archive& archive) {
  archive(cereal::base_class<SequentialLayerConfig>(this), _dim, _sparsity,
          _activation_fn, _sampling_config);
}

template <class Archive>
void EmbeddingLayerConfig::serialize(Archive& archive) {
  archive(_num_embedding_lookups, _lookup_size, _log_embedding_block_size,
          _reduction, _num_tokens_per_input);
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

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedLayerConfig)
