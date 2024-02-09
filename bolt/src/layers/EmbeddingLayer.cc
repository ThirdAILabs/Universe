#include "EmbeddingLayer.h"
#include <hashing/src/MurmurHash.h>
#include <archive/src/Archive.h>
#include <archive/src/ParameterReference.h>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {

std::pair<uint64_t, uint64_t> computeBlockSizeAndNumChunks(
    uint64_t log_block_size, uint64_t lookup_size, uint64_t chunk_size) {
  uint64_t block_size = (1ULL << log_block_size) + lookup_size;
  uint64_t n_chunks = (block_size + chunk_size - 1) / chunk_size;

  return {n_chunks * chunk_size, n_chunks};
}

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint32_t seed)
    : _num_lookups_per_token(config.numEmbeddingLookups()),
      _lookup_size(config.lookupSize()),
      _total_embedding_dim(config.numEmbeddingLookups() * config.lookupSize()),
      _log_embedding_block_size(config.logEmbeddingBlockSize()),
      _update_chunk_size(config.updateChunkSize()),
      _reduction(config.reduction()),
      _num_tokens_per_input(config.numTokensPerInput()),
      _hash_fn(seed),
      _disable_sparse_parameter_updates(false),
      _should_save_optimizer(false) {
  switch (_reduction) {
    case EmbeddingReductionType::SUM:
    case EmbeddingReductionType::AVERAGE:
      break;
    case EmbeddingReductionType::CONCATENATION:
      if (!_num_tokens_per_input) {
        throw std::invalid_argument(
            "Must specify a number of tokens per input with a concatenation "
            "reduction.");
      }
      _total_embedding_dim *= _num_tokens_per_input.value();
      break;
  }

  // We allocate the extra _lookup_size elements such that if a point hashes to
  // the end of 2^_embedding_block_size we don't have to worry about wrapping it
  // around.
  auto [emb_block_size, n_emb_chunks] = computeBlockSizeAndNumChunks(
      _log_embedding_block_size, _lookup_size, _update_chunk_size);

  _embedding_block_size = emb_block_size;

  _embedding_block =
      std::make_shared<std::vector<float>>(_embedding_block_size, 0);

  _embedding_chunks_used = std::vector<bool>(n_emb_chunks, false);

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_embedding_block->begin(), _embedding_block->end(),
                [&]() { return dist(gen); });
}

EmbeddingLayer::EmbeddingLayer(const ar::Archive& archive)
    : _num_lookups_per_token(archive.u64("num_lookups_per_token")),
      _lookup_size(archive.u64("lookup_size")),
      _total_embedding_dim(_num_lookups_per_token * _lookup_size),
      _log_embedding_block_size(archive.u64("log_embedding_block_size")),
      _update_chunk_size(archive.u64("update_chunk_size")),
      _reduction(reductionFromString(archive.str("reduction"))),
      _num_tokens_per_input(archive.getOpt<ar::U64>("num_tokens_per_input")),
      _hash_fn(archive.u64("hash_seed")),
      _embedding_block(archive.get("embeddings")->param().loadedParameter()),
      _disable_sparse_parameter_updates(
          archive.boolean("disable_sparse_parameter_updates")) {
  if (_reduction == EmbeddingReductionType::CONCATENATION) {
    if (!_num_tokens_per_input) {
      throw std::invalid_argument(
          "Must specify a number of tokens per input with a concatenation "
          "reduction.");
    }
    _total_embedding_dim *= _num_tokens_per_input.value();
  }

  auto [emb_block_size, n_emb_chunks] = computeBlockSizeAndNumChunks(
      _log_embedding_block_size, _lookup_size, _update_chunk_size);

  _embedding_block_size = emb_block_size;

  if (_embedding_block->size() != _embedding_block_size) {
    throw std::runtime_error(
        "Embedding block does not have expected size in fromArchive.");
  }

  _embedding_chunks_used.assign(n_emb_chunks, false);

  if (archive.contains("embedding_optimizer")) {
    _optimizer = optimizerFromArchive(*archive.get("embedding_optimizer"));
  }
}

void EmbeddingLayer::forward(const BoltVector& tokens, BoltVector& output) {
  assert(output.len == _total_embedding_dim);
  assert(_reduction != EmbeddingReductionType::CONCATENATION ||
         _num_tokens_per_input.value() == tokens.len);
  assert(output.active_neurons == nullptr);

  if (_reduction != EmbeddingReductionType::CONCATENATION) {
    std::fill_n(output.activations, _total_embedding_dim, 0);
  }
  if (tokens.isDense()) {
    throw std::invalid_argument(
        "Cannot pass dense BoltVector as tokens in EmbeddingLayer.");
  }

  std::fill_n(output.gradients, _total_embedding_dim, 0);

  // Preform outer dereferencing once here to avoid repeating it later.
  auto& embedding_block = *_embedding_block;

  for (uint64_t lookup_index = 0; lookup_index < _num_lookups_per_token;
       lookup_index++) {
    float* output_start =
        output.activations + getOutputOffsetWithinEmbedding(lookup_index);

    for (uint64_t token_idx = 0; token_idx < tokens.len; token_idx++) {
      uint32_t token = tokens.active_neurons[token_idx];
      uint64_t embedding_block_offset =
          getEmbeddingBlockOffset(token, lookup_index);

      assert(embedding_block_offset < _embedding_block_size - _lookup_size);

      switch (_reduction) {
        case EmbeddingReductionType::SUM:
        case EmbeddingReductionType::AVERAGE:
          // Safe since we allocated 2^_log_embedding_block_size+_lookup_size
          for (uint64_t i = 0; i < _lookup_size; i++) {
            output_start[i] += embedding_block[embedding_block_offset + i];
          }
          break;
        case EmbeddingReductionType::CONCATENATION:
          std::copy(
              embedding_block.data() + embedding_block_offset,
              embedding_block.data() + embedding_block_offset + _lookup_size,
              output_start);

          // Shift output_start since each token maps to unique range in the
          // output vector.
          output_start += _num_lookups_per_token * _lookup_size;
          break;
      }
    }

    if (_reduction == EmbeddingReductionType::AVERAGE) {
      for (uint32_t i = 0; i < _lookup_size; i++) {
        output_start[i] /= tokens.len;
      }
    }
  }
}

void EmbeddingLayer::backpropagate(const BoltVector& tokens,
                                   const BoltVector& output) {
  for (uint64_t lookup_index = 0; lookup_index < _num_lookups_per_token;
       lookup_index++) {
    const float* output_gradients =
        output.gradients + getOutputOffsetWithinEmbedding(lookup_index);

    for (uint64_t token_index = 0; token_index < tokens.len; token_index++) {
      uint32_t token = tokens.active_neurons[token_index];
      uint64_t embedding_block_offset =
          getEmbeddingBlockOffset(token, lookup_index);
      markUsedChunks(embedding_block_offset);

      assert(embedding_block_offset < _embedding_block_size - _lookup_size);

      float* update_loc = _optimizer->gradients.data() + embedding_block_offset;

      if (_reduction == EmbeddingReductionType::AVERAGE) {
        for (uint32_t i = 0; i < _lookup_size; i++) {
          update_loc[i] += output_gradients[i] / tokens.len;
        }
      } else {
        for (uint32_t i = 0; i < _lookup_size; i++) {
          update_loc[i] += output_gradients[i];
        }
      }

      if (_reduction == EmbeddingReductionType::CONCATENATION) {
        // Shift output_gradients since each token maps to unique range in the
        // output vector.
        output_gradients += _num_lookups_per_token * _lookup_size;
      }
    }
  }
}

void EmbeddingLayer::updateParameters(float lr, uint32_t iter, float B1,
                                      float B2, float eps) {
  if (_disable_sparse_parameter_updates) {
    _optimizer->applyUpdate(*_embedding_block, lr, iter);
  } else {
    updateParametersSparse(lr, iter, B1, B2, eps);
  }
}

void EmbeddingLayer::updateParametersSparse(float lr, uint32_t iter, float B1,
                                            float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

  // Preform outer dereferencing once here to avoid repeating it later.
  auto& embedding_block = *_embedding_block;

#pragma omp parallel for default(none) shared( \
    embedding_block, B1, B2, B1_bias_corrected, B2_bias_corrected, eps, lr)
  for (uint64_t chunk_id = 0; chunk_id < _embedding_chunks_used.size();
       chunk_id++) {
    if (!_embedding_chunks_used[chunk_id]) {
      continue;
    }

    _embedding_chunks_used[chunk_id] = false;

    for (uint64_t n = chunk_id * _update_chunk_size;
         n < (chunk_id + 1) * _update_chunk_size; n++) {
      float grad = _optimizer->gradients[n];
      if (grad == 0.0) {
        // Because the chunk being updated may not have entirely been used we
        // check for this to avoid updating unused elements of the embedding
        // table. It is highly unlikely that the gradient would be zero if the
        // section of the embedding table was used.
        continue;
      }
      assert(!std::isnan(grad));

      _optimizer->momentum[n] = B1 * _optimizer->momentum[n] + (1 - B1) * grad;
      _optimizer->velocity[n] =
          B2 * _optimizer->velocity[n] + (1 - B2) * grad * grad;
      assert(!std::isnan(_optimizer->momentum[n]));
      assert(!std::isnan(_optimizer->velocity[n]));

      embedding_block[n] +=
          lr * (_optimizer->momentum[n] / B1_bias_corrected) /
          (std::sqrt(_optimizer->velocity[n] / B2_bias_corrected) + eps);
      assert(!std::isnan(embedding_block[n]));

      _optimizer->gradients[n] = 0;
    }
  }
}

void EmbeddingLayer::buildLayerSummary(std::ostream& summary) const {
  summary << " num_embedding_lookups=" << _num_lookups_per_token;
  summary << ", lookup_size=" << _lookup_size;
  summary << ", log_embedding_block_size=" << _log_embedding_block_size;
  summary << ", reduction=" << reduction();

  if (_num_tokens_per_input) {
    summary << ", num_tokens_per_input=" << _num_tokens_per_input.value();
  }
}

std::unique_ptr<EmbeddingLayer> EmbeddingLayer::duplicateWithNewReduction(
    const std::string& reduction,
    std::optional<uint64_t> num_tokens_per_input) const {
  EmbeddingLayerConfig config(_num_lookups_per_token, _lookup_size,
                              _log_embedding_block_size, _update_chunk_size,
                              reduction, num_tokens_per_input);

  auto new_layer = std::make_unique<EmbeddingLayer>(config);

  if (new_layer->_embedding_block_size != _embedding_block_size) {
    throw std::runtime_error(
        "Expected embedding_block_size to be consistent in "
        "duplicateWithNewReduction.");
  }
  new_layer->_hash_fn = _hash_fn;
  new_layer->_embedding_block = _embedding_block;

  return new_layer;
}

}  // namespace thirdai::bolt