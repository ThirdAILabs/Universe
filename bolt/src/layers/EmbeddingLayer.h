#pragma once

#include "BoltVector.h"
#include "LayerConfig.h"
#include <hashing/src/UniversalHash.h>
#include <cmath>
#include <ctime>
#include <vector>

namespace thirdai::bolt {

namespace tests {
class EmbeddingLayerTestFixture;
}  // namespace tests

class EmbeddingLayer {
  friend class tests::EmbeddingLayerTestFixture;

 public:
  explicit EmbeddingLayer(const EmbeddingLayerConfig& config,
                          uint32_t seed = time(nullptr));

  void forward(uint32_t vec_index, const std::vector<uint32_t>& tokens,
               BoltVector& output);

  void backpropagate(uint32_t vec_index, const BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  uint32_t getEmbeddingDim() const { return _total_embedding_dim_bytes; }

  void initializeLayer(uint32_t new_batch_size);

  BoltBatch createBatchState(const uint32_t batch_size) const {
    return BoltBatch(_total_embedding_dim_bytes, batch_size, true);
  }

  void buildLayerSummary(std::stringstream& summary) const;

  EmbeddingLayer(const EmbeddingLayer&) = delete;
  EmbeddingLayer(EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&&) = delete;

  ~EmbeddingLayer() = default;

 private:
  std::vector<std::pair<uint64_t, uint64_t>> getDisjointUpdateRanges() const;

  inline uint32_t getHashLocForToken(uint32_t token, uint32_t lookup_index) {
    uint64_t id = token * _num_lookups_per_token + lookup_index;
    uint32_t hash = _hash_fn.gethash(id);

    // We bit shift to make sure that the hash loc is within the range of the
    // embedding block.
    return hash >> (32 - _log_embedding_block_size);
  }

  constexpr uint32_t getOutputOffsetWithinEmbedding(
      uint32_t lookup_index) const {
    return lookup_index * _lookup_size_bytes;
  }

  static constexpr uint32_t getHashLocIndex(uint32_t lookup_index,
                                            uint32_t token_index,
                                            uint32_t num_tokens) {
    return lookup_index * num_tokens + token_index;
  }

  void recordHashLoc(uint32_t vec_index, uint64_t hash_loc) {
    _hash_locs[vec_index].push_back(hash_loc);
  }

  uint64_t retrieveHashLoc(uint32_t vec_index, uint32_t lookup_index,
                      uint32_t token_index, uint32_t num_tokens) {
    return _hash_locs[vec_index]
                     [getHashLocIndex(lookup_index, token_index, num_tokens)];
  }

  uint32_t _num_lookups_per_token, _lookup_size_bytes,
      _total_embedding_dim_bytes, _log_embedding_block_size;
  uint64_t _embedding_block_size_bytes;

  hashing::UniversalHash _hash_fn;

  std::vector<float> _embedding_block;
  std::vector<float> _gradients;
  std::vector<float> _momentum;
  std::vector<float> _velocity;

  std::vector<std::vector<uint64_t>> _hash_locs;
};

}  // namespace thirdai::bolt