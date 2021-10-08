#include "EmbeddingLayer.h"
#include "../../utils/hashing/MurmurHash.h"
#include <algorithm>
#include <random>

namespace thirdai::bolt {

EmbeddingLayer::EmbeddingLayer(uint32_t num_embedding_lookups,
                               uint32_t lookup_size,
                               uint32_t log_embedding_block_size, uint32_t seed)
    : _num_embedding_lookups(num_embedding_lookups),
      _lookup_size(lookup_size),
      _total_embedding_dim(num_embedding_lookups * lookup_size),
      _log_embedding_block_size(log_embedding_block_size),
      _batch_size(0),
      _embeddings(nullptr),
      _errors(nullptr),
      _lens(nullptr),
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

void EmbeddingLayer::feedForward(uint32_t batch_indx, const uint32_t* tokens,
                                 uint32_t len) {
  _lens[batch_indx] = len;
  delete[] _embedding_locs[batch_indx];
  _embedding_locs[batch_indx] = new uint32_t[len * _num_embedding_lookups];

  std::fill_n(_embeddings[batch_indx], _total_embedding_dim, 0);
  std::fill_n(_errors[batch_indx], _total_embedding_dim, 0);

  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    float* output_start = _embeddings[batch_indx] + e * _lookup_size;

    for (uint32_t n = 0; n < len; n++) {
      uint32_t id = tokens[n] * _num_embedding_lookups + e;
      uint32_t hash_loc = utils::MurmurHash(reinterpret_cast<const char*>(&id),
                                            sizeof(uint32_t), _seed);
      hash_loc = hash_loc >> (32 - _log_embedding_block_size);
      _embedding_locs[batch_indx][n * _num_embedding_lookups + e] = hash_loc;

      // This is safe since we allowed 2^_log_embedding_block_size+_lookup_size
      for (uint32_t i = 0; i < _lookup_size; i++) {
        output_start[i] += _embedding_block[hash_loc + i];
      }
    }
  }
}

void EmbeddingLayer::backpropagate(uint32_t batch_indx, float learning_rate) {
  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    const float* errors = _errors[batch_indx] + e * _lookup_size;

    for (uint32_t n = 0; n < _lens[batch_indx]; n++) {
      float* update_loc =
          _embedding_block +
          _embedding_locs[batch_indx][n * _num_embedding_lookups + e];

      for (uint32_t i = 0; i < _lookup_size; i++) {
        update_loc[i] += learning_rate * errors[i];
      }
    }
  }
}

void EmbeddingLayer::setBatchSize(uint32_t new_batch_size) {
  if (new_batch_size == _batch_size) {
    return;
  }

  for (uint32_t b = 0; b < _batch_size; b++) {
    delete[] _embeddings[b];
    delete[] _errors[b];
    delete[] _embedding_locs[b];
  }

  delete[] _embeddings;
  delete[] _errors;
  delete[] _embedding_locs;
  delete[] _lens;

  _batch_size = new_batch_size;

  _embeddings = new float*[_batch_size];
  _errors = new float*[_batch_size];
  _embedding_locs = new uint32_t*[_batch_size];
  _lens = new uint32_t[_batch_size];

  for (uint32_t b = 0; b < _batch_size; b++) {
    _embeddings[b] = new float[_total_embedding_dim]();
    _errors[b] = new float[_total_embedding_dim]();
    _embedding_locs[b] = nullptr;
  }
}

EmbeddingLayer::~EmbeddingLayer() {
  delete[] _embedding_block;

  for (uint32_t b = 0; b < _batch_size; b++) {
    delete[] _embeddings[b];
    delete[] _errors[b];
    delete[] _embedding_locs[b];
  }

  delete[] _embeddings;
  delete[] _errors;
  delete[] _embedding_locs;
  delete[] _lens;
}

}  // namespace thirdai::bolt