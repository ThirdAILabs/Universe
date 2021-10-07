#pragma once

#include "Layer.h"
#include <ctime>

namespace thirdai::bolt {

namespace tests {
class EmbeddingLayerTestFixture;
}  // namespace tests

class EmbeddingLayer {
  friend class tests::EmbeddingLayerTestFixture;

 public:
  EmbeddingLayer(uint32_t num_embedding_lookups, uint32_t lookup_size,
                 uint32_t log_embedding_block_size,
                 uint32_t seed = time(nullptr));

  void FeedForward(uint32_t batch_indx, const uint32_t* tokens, uint32_t len);

  void Backpropagate(uint32_t batch_indx, float learning_rate);

  uint32_t GetLen(uint32_t /* unused */) const { return _total_embedding_dim; }

  const float* GetEmbedding(uint32_t batch_indx) const {
    return _embeddings[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) { return _errors[batch_indx]; }

  void SetBatchSize(uint32_t new_batch_size);

  EmbeddingLayer(EmbeddingLayer&) = delete;
  EmbeddingLayer(const EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&&) = delete;

  ~EmbeddingLayer();

 private:
  uint32_t _num_embedding_lookups, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size, _embedding_block_size, _batch_size, _seed;

  float* _embedding_block;

  float** _embeddings;
  float** _errors;

  uint32_t* _lens;
  uint32_t** _embedding_locs;
};

}  // namespace thirdai::bolt