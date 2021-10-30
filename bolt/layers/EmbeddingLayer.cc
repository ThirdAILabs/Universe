#include "EmbeddingLayer.h"
#include "../../utils/hashing/MurmurHash.h"
#include <algorithm>
#include <random>
#include <vector>

namespace thirdai::bolt {

EmbeddingLayer::EmbeddingLayer(const EmbeddingLayerConfig& config,
                               uint32_t seed)
    : _num_embedding_lookups(config.num_embedding_lookups),
      _lookup_size(config.lookup_size),
      _total_embedding_dim(config.num_embedding_lookups * config.lookup_size),
      _log_embedding_block_size(config.log_embedding_block_size),
      _batch_size(0),
      _embeddings(nullptr),
      _errors(nullptr),
      _internal_state_provided(false),
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
  _loc_lens[batch_indx] = len;
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

void EmbeddingLayer::backpropagate(uint32_t batch_indx) {
  for (uint32_t e = 0; e < _num_embedding_lookups; e++) {
    const float* errors = _errors[batch_indx] + e * _lookup_size;

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

void EmbeddingLayer::updateParameters(float lr, uint32_t iter, float B1,
                                      float B2, float eps) {
  float B1_ = static_cast<float>(1 - pow(B1, iter));
  float B2_ = static_cast<float>(1 - pow(B2, iter));

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
#pragma omp parallel for default(none) \
    shared(disjoint_ranges, B1, B2, B1_, B2_, eps, lr)
  for (const auto& pair : disjoint_ranges) {
    for (uint64_t n = pair.first; n < pair.second; n++) {
      float grad = _gradients[n];
      _momentum[n] = B1 * _momentum[n] + (1 - B1) * grad;
      _velocity[n] = B2 * _velocity[n] + (1 - B2) * grad * grad;

      _embedding_block[n] +=
          lr * (_momentum[n] / B1_) / (std::sqrt(_velocity[n] / B2_) + eps);

      _gradients[n] = 0;
    }
  }
}

void EmbeddingLayer::initializeLayer(uint32_t new_batch_size) {
  if (new_batch_size <= _batch_size) {
    return;
  }

  deallocateInternalState();

  _batch_size = new_batch_size;
  _internal_state_provided = false;

  _embeddings = new float*[_batch_size];
  _errors = new float*[_batch_size];
  _embedding_locs = new uint32_t*[_batch_size];
  _loc_lens = new uint32_t[_batch_size];

  for (uint32_t b = 0; b < _batch_size; b++) {
    _embeddings[b] = new float[_total_embedding_dim]();
    _errors[b] = new float[_total_embedding_dim]();
    _embedding_locs[b] = nullptr;
  }
}

void EmbeddingLayer::initializeLayer(uint32_t batch_size,
                                     float** new_embeddings,
                                     float** new_errors) {
  deallocateInternalState();

  _batch_size = batch_size;
  _internal_state_provided = true;

  _embeddings = new_embeddings;
  _errors = new_errors;
  _loc_lens = new uint32_t[_batch_size];
  _embedding_locs = new uint32_t*[_batch_size];

  for (uint32_t b = 0; b < _batch_size; b++) {
    _embedding_locs[b] = nullptr;
  }
}

void EmbeddingLayer::deallocateInternalState() {
  for (uint32_t b = 0; b < _batch_size; b++) {
    if (!_internal_state_provided) {
      delete[] _embeddings[b];
      delete[] _errors[b];
    }
    delete[] _embedding_locs[b];
  }

  delete[] _embeddings;
  delete[] _errors;
  delete[] _embedding_locs;
  delete[] _loc_lens;
}

EmbeddingLayer::~EmbeddingLayer() {
  delete[] _embedding_block;
  delete[] _gradients;
  delete[] _momentum;
  delete[] _velocity;

  deallocateInternalState();
}

}  // namespace thirdai::bolt