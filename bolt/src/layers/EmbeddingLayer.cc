#include "EmbeddingLayer.h"
#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <random>
#include <stdexcept>

namespace thirdai::bolt {

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint32_t seed)
    : _num_lookups_per_token(config.numEmbeddingLookups()),
      _lookup_size(config.lookupSize()),
      _total_embedding_dim(config.numEmbeddingLookups() * config.lookupSize()),
      _log_embedding_block_size(config.logEmbeddingBlockSize()),
      _embedding_chunk_size(config.embeddingChunkSize()),
      _reduction(config.reduction()),
      _num_tokens_per_input(config.numTokensPerInput()),
      _hash_fn(seed),
      _disable_sparse_parameter_updates(false) {
  switch (_reduction) {
    case EmbeddingReductionType::SUM:
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
  _embedding_block_size = (1 << _log_embedding_block_size) + _lookup_size;
  uint64_t n_chunks = (_embedding_block_size + _embedding_chunk_size - 1) /
                      _embedding_chunk_size;
  _embedding_block_size = n_chunks * _embedding_chunk_size;
  _embedding_block = std::vector<float>(_embedding_block_size, 0);

  initOptimizer();

  _embedding_chunks_used = std::vector<bool>(n_chunks, false);

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_embedding_block.begin(), _embedding_block.end(),
                [&]() { return dist(gen); });
}

void EmbeddingLayer::forward(const BoltVector& tokens, BoltVector& output) {
  assert(output.len == _total_embedding_dim);
  assert(_reduction == EmbeddingReductionType::SUM ||
         _num_tokens_per_input.value() == tokens.len);
  assert(output.active_neurons == nullptr);

  if (_reduction == EmbeddingReductionType::SUM) {
    std::fill_n(output.activations, _total_embedding_dim, 0);
  }
  if (tokens.isDense()) {
    throw std::invalid_argument(
        "Cannot pass dense BoltVector as tokens in EmbeddingLayer.");
  }

  std::fill_n(output.gradients, _total_embedding_dim, 0);

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
          // Safe since we allocated 2^_log_embedding_block_size+_lookup_size
          for (uint64_t i = 0; i < _lookup_size; i++) {
            output_start[i] += _embedding_block[embedding_block_offset + i];
          }
          break;
        case EmbeddingReductionType::CONCATENATION:
          std::copy(
              _embedding_block.data() + embedding_block_offset,
              _embedding_block.data() + embedding_block_offset + _lookup_size,
              output_start);

          // Shift output_start since each token maps to unique range in the
          // output vector.
          output_start += _num_lookups_per_token * _lookup_size;
          break;
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

      for (uint64_t i = 0; i < _lookup_size; i++) {
        update_loc[i] += output_gradients[i];
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
    updateParametersDense(lr, iter, B1, B2, eps);
  } else {
    updateParametersSparse(lr, iter, B1, B2, eps);
  }
}

void EmbeddingLayer::updateParametersSparse(float lr, uint32_t iter, float B1,
                                            float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

#pragma omp parallel for default(none) \
    shared(B1, B2, B1_bias_corrected, B2_bias_corrected, eps, lr)
  for (uint64_t chunk_id = 0; chunk_id < _embedding_chunks_used.size();
       chunk_id++) {
    if (!_embedding_chunks_used[chunk_id]) {
      continue;
    }

    _embedding_chunks_used[chunk_id] = false;

    for (uint64_t n = chunk_id * _embedding_chunk_size;
         n < (chunk_id + 1) * _embedding_chunk_size; n++) {
      float grad = _optimizer->gradients[n];
      assert(!std::isnan(grad));

      _optimizer->momentum[n] = B1 * _optimizer->momentum[n] + (1 - B1) * grad;
      _optimizer->velocity[n] =
          B2 * _optimizer->velocity[n] + (1 - B2) * grad * grad;
      assert(!std::isnan(_optimizer->momentum[n]));
      assert(!std::isnan(_optimizer->velocity[n]));

      _embedding_block[n] +=
          lr * (_optimizer->momentum[n] / B1_bias_corrected) /
          (std::sqrt(_optimizer->velocity[n] / B2_bias_corrected) + eps);
      assert(!std::isnan(_embedding_block[n]));

      _optimizer->gradients[n] = 0;
    }
  }
}

void EmbeddingLayer::updateParametersDense(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

#pragma omp parallel for default(none) \
    shared(B1, B2, B1_bias_corrected, B2_bias_corrected, eps, lr)
  for (uint64_t n = 0; n < _embedding_block_size; n++) {
    float grad = _optimizer->gradients[n];
    assert(!std::isnan(grad));

    _optimizer->momentum[n] = B1 * _optimizer->momentum[n] + (1 - B1) * grad;
    _optimizer->velocity[n] =
        B2 * _optimizer->velocity[n] + (1 - B2) * grad * grad;
    assert(!std::isnan(_optimizer->momentum[n]));
    assert(!std::isnan(_optimizer->velocity[n]));

    _embedding_block[n] +=
        lr * (_optimizer->momentum[n] / B1_bias_corrected) /
        (std::sqrt(_optimizer->velocity[n] / B2_bias_corrected) + eps);
    assert(!std::isnan(_embedding_block[n]));

    _optimizer->gradients[n] = 0;
  }
}

void EmbeddingLayer::buildLayerSummary(std::stringstream& summary) const {
  summary << " num_embedding_lookups=" << _num_lookups_per_token;
  summary << ", lookup_size=" << _lookup_size;
  summary << ", log_embedding_block_size=" << _log_embedding_block_size;
  switch (_reduction) {
    case EmbeddingReductionType::SUM:
      summary << ", reduction=sum";
      break;
    case EmbeddingReductionType::CONCATENATION:
      summary << ", reduction=concatenation";
  }
  if (_num_tokens_per_input) {
    summary << ", num_tokens_per_input=" << _num_tokens_per_input.value();
  }
  summary << "\n";
}

}  // namespace thirdai::bolt