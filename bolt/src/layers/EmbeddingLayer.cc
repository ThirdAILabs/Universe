#include "EmbeddingLayer.h"
#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <random>

namespace thirdai::bolt {

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint32_t seed)
    : _num_lookups_per_token(config.num_embedding_lookups),
      _lookup_size_bytes(config.lookup_size),
      _total_embedding_dim_bytes(config.num_embedding_lookups *
                                 config.lookup_size),
      _log_embedding_block_size(config.log_embedding_block_size),
      _hash_fn(seed) {
  // We allocate the extra _lookup_size elements such that if a point hashes to
  // the end of 2^_embedding_block_size we don't have to worry about wrapping it
  // around.
  _embedding_block_size_bytes =
      (1 << _log_embedding_block_size) + _lookup_size_bytes;
  _embedding_block = std::vector<float>(_embedding_block_size_bytes, 0);
  _gradients = std::vector<float>(_embedding_block_size_bytes, 0);
  _momentum = std::vector<float>(_embedding_block_size_bytes, 0);
  _velocity = std::vector<float>(_embedding_block_size_bytes, 0);

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_embedding_block.begin(), _embedding_block.end(),
                [&]() { return dist(gen); });
}

void EmbeddingLayer::forward(uint32_t vec_index,
                             const std::vector<uint32_t>& tokens,
                             BoltVector& output) {
  assert(output.len == _total_embedding_dim);
  assert(output.active_neurons == nullptr);

  std::fill_n(output.activations, _total_embedding_dim_bytes, 0);
  std::fill_n(output.gradients, _total_embedding_dim_bytes, 0);

  _hash_locs[vec_index].clear();
  _hash_locs[vec_index].reserve(tokens.size() * _num_lookups_per_token);

  for (uint32_t lookup_index = 0; lookup_index < _num_lookups_per_token;
       lookup_index++) {
    float* output_start =
        output.activations + getOutputOffsetWithinEmbedding(lookup_index);

    for (uint32_t token : tokens) {
      uint64_t hash_loc = getHashLocForToken(token, lookup_index);
      recordHashLoc(vec_index, hash_loc);

      assert(hash_loc < _embedding_block_size - _lookup_size);

      // Safe since we allocated 2^_log_embedding_block_size+_lookup_size
      for (uint32_t i = 0; i < _lookup_size_bytes; i++) {
        output_start[i] += _embedding_block[hash_loc + i];
      }
    }
  }
}

void EmbeddingLayer::backpropagate(uint32_t vec_index,
                                   const BoltVector& output) {
  uint32_t num_tokens = _hash_locs[vec_index].size() / _num_lookups_per_token;

  for (uint32_t lookup_index = 0; lookup_index < _num_lookups_per_token;
       lookup_index++) {
    const float* output_gradients =
        output.gradients + getOutputOffsetWithinEmbedding(lookup_index);

    for (uint32_t token_index = 0; token_index < num_tokens; token_index++) {
      uint64_t hash_loc =
          retrieveHashLoc(vec_index, lookup_index, token_index, num_tokens);

      assert(hash_loc < _embedding_block_size - _lookup_size);

      float* update_loc = _gradients.data() + hash_loc;

      for (uint32_t i = 0; i < _lookup_size_bytes; i++) {
        update_loc[i] += output_gradients[i];
      }
    }
  }
}

void EmbeddingLayer::updateParameters(float lr, uint32_t iter, float B1,
                                      float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

  std::vector<std::pair<uint64_t, uint64_t>> disjoint_ranges =
      getDisjointUpdateRanges();

#pragma omp parallel for default(none) shared( \
    disjoint_ranges, B1, B2, B1_bias_corrected, B2_bias_corrected, eps, lr)
  for (uint32_t pair_id = 0; pair_id < disjoint_ranges.size();  // NOLINT
       pair_id++) {
    // MSVC doesn't like if we iterate over objects, only integers
    // (but clang-tidy wants the range based for loop, so we need NOLINT above)
    const auto& pair = disjoint_ranges[pair_id];
    for (uint64_t n = pair.first; n < pair.second; n++) {
      float grad = _gradients[n];
      assert(!std::isnan(grad));

      _momentum[n] = B1 * _momentum[n] + (1 - B1) * grad;
      _velocity[n] = B2 * _velocity[n] + (1 - B2) * grad * grad;
      assert(!std::isnan(_momentum[n]));
      assert(!std::isnan(_velocity[n]));

      _embedding_block[n] +=
          lr * (_momentum[n] / B1_bias_corrected) /
          (std::sqrt(_velocity[n] / B2_bias_corrected) + eps);
      assert(!std::isnan(_embedding_block[n]));

      _gradients[n] = 0;
    }
  }
}

void EmbeddingLayer::initializeLayer(uint32_t new_batch_size) {
  _hash_locs = std::vector<std::vector<uint64_t>>(new_batch_size);
}

void EmbeddingLayer::buildLayerSummary(std::stringstream& summary) const {
  summary << " num_embedding_lookups=" << _num_lookups_per_token;
  summary << ", lookup_size=" << _lookup_size_bytes;
  summary << ", log_embedding_block_size=" << _log_embedding_block_size;
}

std::vector<std::pair<uint64_t, uint64_t>>
EmbeddingLayer::getDisjointUpdateRanges() const {
  std::vector<uint64_t> all_embedding_locs;
  for (const auto& locs : _hash_locs) {
    all_embedding_locs.insert(all_embedding_locs.end(), locs.begin(),
                              locs.end());
  }

  std::sort(all_embedding_locs.begin(), all_embedding_locs.end());

  std::vector<std::pair<uint64_t, uint64_t>> disjoint_ranges;
  for (uint32_t i = 0; i < all_embedding_locs.size(); i++) {
    uint64_t start = all_embedding_locs[i];
    uint64_t end = start + _lookup_size_bytes;
    for (uint32_t j = i + 1; j < all_embedding_locs.size(); j++) {
      if (all_embedding_locs[j] > end) {
        break;
      }
      end = all_embedding_locs[j] + _lookup_size_bytes;
      i++;
    }
    disjoint_ranges.push_back({start, end});
  }

  return disjoint_ranges;
}

}  // namespace thirdai::bolt