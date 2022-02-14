#include "EmbeddingLayer.h"
#include <hashing/src/MurmurHash.h>
#include <algorithm>
#include <random>

namespace thirdai::bolt {

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint32_t seed)
    : _num_embedding_lookups(config.num_embedding_lookups),
      _lookup_size(config.lookup_size),
      _total_embedding_dim(config.num_embedding_lookups * config.lookup_size),
      _log_embedding_block_size(config.log_embedding_block_size),
      _batch_size(0),
      _loc_lens(nullptr),
      _embedding_locs(nullptr) {
  // We allocate the extra _lookup_size elements such that if a point hashes to
  // the end of 2^_embedding_block_size we don't have to worry about wrapping it
  // around.
  _embedding_block_size = (1 << _log_embedding_block_size) + _lookup_size;
  _embedding_block = new float[_embedding_block_size];
  _gradients = new float[_embedding_block_size]();
  _momentum = new float[_embedding_block_size]();
  _velocity = new float[_embedding_block_size]();
  assert(_embedding_block != nullptr);
  assert(_gradients != nullptr);
  assert(_momentum != nullptr);
  assert(_velocity != nullptr);

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_embedding_block, _embedding_block + _embedding_block_size,
                [&]() { return dist(gen); });

  std::uniform_int_distribution<uint32_t> int_dist(
      0, std::numeric_limits<uint32_t>::max());

  _seed = int_dist(gen);
}

void EmbeddingLayer::forward(uint32_t batch_indx,
                             const std::vector<uint32_t>& tokens,
                             BoltVector& output) {
  assert(output.len == _total_embedding_dim);
  assert(output.active_neurons == nullptr);

  _loc_lens[batch_indx] = tokens.size();
  delete[] _embedding_locs[batch_indx];
  _embedding_locs[batch_indx] =
      new uint32_t[tokens.size() * _num_embedding_lookups];

  std::fill_n(output.activations, _total_embedding_dim, 0);
  std::fill_n(output.gradients, _total_embedding_dim, 0);

  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    float* output_start = output.activations + e * _lookup_size;

    for (uint32_t n = 0; n < tokens.size(); n++) {
      uint32_t id = tokens[n] * _num_embedding_lookups + e;
      uint32_t hash_loc = hashing::MurmurHash(
          reinterpret_cast<const char*>(&id), sizeof(uint32_t), _seed);
      hash_loc = hash_loc >> (32 - _log_embedding_block_size);
      assert(hash_loc < _total_embedding_dim);
      _embedding_locs[batch_indx][n * _num_embedding_lookups + e] = hash_loc;

      // Safe since we allocated 2^_log_embedding_block_size+_lookup_size
      for (uint32_t i = 0; i < _lookup_size; i++) {
        output_start[i] += _embedding_block[hash_loc + i];
      }
    }
  }
}

void EmbeddingLayer::backpropagate(uint32_t batch_indx,
                                   const BoltVector& output) {
  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    const float* errors = output.gradients + e * _lookup_size;

    for (uint32_t n = 0; n < _loc_lens[batch_indx]; n++) {
      float* update_loc =
          _gradients +
          _embedding_locs[batch_indx][n * _num_embedding_lookups + e];

      for (uint32_t i = 0; i < _lookup_size; i++) {
        update_loc[i] += errors[i];
      }
    }
  }
}

std::vector<std::pair<uint64_t, uint64_t>>
EmbeddingLayer::getDisjointUpdateRanges() {
  std::vector<uint32_t> all_embedding_locs;
  for (uint32_t b = 0; b < _batch_size; b++) {
    for (uint32_t n = 0; n < _loc_lens[b]; n++) {
      for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
        all_embedding_locs.push_back(
            _embedding_locs[b][n * _num_embedding_lookups + e]);
      }
    }
  }

  std::sort(all_embedding_locs.begin(), all_embedding_locs.end());

  std::vector<std::pair<uint64_t, uint64_t>> disjoint_ranges;
  for (uint32_t i = 0; i < all_embedding_locs.size(); i++) {
    uint64_t start = all_embedding_locs[i];
    uint64_t end = start + _lookup_size;
    for (uint32_t j = i + 1; j < all_embedding_locs.size(); j++) {
      if (all_embedding_locs[j] > end) {
        break;
      }
      end = all_embedding_locs[j] + _lookup_size;
      i++;
    }
    disjoint_ranges.push_back({start, end});
  }

  return disjoint_ranges;
}

void EmbeddingLayer::updateParameters(float lr, uint32_t iter, float B1,
                                      float B2, float eps) {
  float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
  float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

  std::vector<std::pair<uint64_t, uint64_t>> disjoint_ranges =
      getDisjointUpdateRanges();

#pragma omp parallel for default(none) shared( \
    disjoint_ranges, B1, B2, B1_bias_corrected, B2_bias_corrected, eps, lr)
  for (const auto& pair : disjoint_ranges) {
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
  if (new_batch_size <= _batch_size) {
    return;
  }
  for (uint32_t b = 0; b < _batch_size; b++) {
    delete[] _embedding_locs[b];
  }

  delete[] _embedding_locs;
  delete[] _loc_lens;

  _batch_size = new_batch_size;

  _loc_lens = new uint32_t[_batch_size];
  assert(_loc_lens != nullptr);
  _embedding_locs = new uint32_t*[_batch_size]();
  assert(_embedding_locs != nullptr);
}

EmbeddingLayer::~EmbeddingLayer() {
  delete[] _embedding_block;
  delete[] _gradients;
  delete[] _momentum;
  delete[] _velocity;

  for (uint32_t b = 0; b < _batch_size; b++) {
    delete[] _embedding_locs[b];
  }

  delete[] _embedding_locs;
  delete[] _loc_lens;
}

}  // namespace thirdai::bolt