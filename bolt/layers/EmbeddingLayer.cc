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

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0, 0.01);

  std::generate(_embedding_block, _embedding_block + _embedding_block_size,
                [&]() { return dist(gen); });

  std::uniform_int_distribution<uint32_t> int_dist(
      0, std::numeric_limits<uint32_t>::max());

  _seed = int_dist(gen);
}

void EmbeddingLayer::forward(uint32_t batch_indx, const uint32_t* tokens,
                             uint32_t len, VectorState& output) {
  _loc_lens[batch_indx] = len;
  delete[] _embedding_locs[batch_indx];
  _embedding_locs[batch_indx] = new uint32_t[len * _num_embedding_lookups];

  std::fill_n(output.activations, _total_embedding_dim, 0);
  std::fill_n(output.gradients, _total_embedding_dim, 0);

  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    float* output_start = output.activations + e * _lookup_size;

    for (uint32_t n = 0; n < len; n++) {
      uint32_t id = tokens[n] * _num_embedding_lookups + e;
      uint32_t hash_loc = hashing::MurmurHash(
          reinterpret_cast<const char*>(&id), sizeof(uint32_t), _seed);
      hash_loc = hash_loc >> (32 - _log_embedding_block_size);
      _embedding_locs[batch_indx][n * _num_embedding_lookups + e] = hash_loc;

      // This is safe since we allowed 2^_log_embedding_block_size+_lookup_size
      for (uint32_t i = 0; i < _lookup_size; i++) {
        output_start[i] += _embedding_block[hash_loc + i];
      }
    }
  }
}

void EmbeddingLayer::backpropagate(uint32_t batch_indx, float learning_rate,
                                   const VectorState& output) {
  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    const float* errors = output.gradients + e * _lookup_size;

    for (uint32_t n = 0; n < _loc_lens[batch_indx]; n++) {
      float* update_loc =
          _embedding_block +
          _embedding_locs[batch_indx][n * _num_embedding_lookups + e];

      for (uint32_t i = 0; i < _lookup_size; i++) {
        update_loc[i] += learning_rate * errors[i];
      }
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
  _embedding_locs = new uint32_t*[_batch_size]();
}

EmbeddingLayer::~EmbeddingLayer() {
  delete[] _embedding_block;

  for (uint32_t b = 0; b < _batch_size; b++) {
    delete[] _embedding_locs[b];
  }

  delete[] _embedding_locs;
  delete[] _loc_lens;
}

}  // namespace thirdai::bolt