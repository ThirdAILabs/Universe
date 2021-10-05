#pragma once

#include "Layer.h"
#include <time.h>

namespace thirdai::bolt {

class EmbeddingLayer {
 public:
  EmbeddingLayer(uint32_t num_embedding_lookups,
                 uint32_t log_embedding_block_size, uint32_t lookup_size,
                 uint32_t seed = time(nullptr));

  void FeedForward(uint32_t batch_indx, const uint32_t* items, uint32_t len);

  void Backpropagate(uint32_t batch_indx);

  void UpdateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  uint32_t GetLen(uint32_t batch_indx) const { return _total_embedding_dim; }

  const float* GetValues(uint32_t batch_indx) const {
    return _embeddings[batch_indx];
  }

  float* GetErrors(uint32_t batch_indx) { return _errors[batch_indx]; }

  void SetBatchSize(uint32_t new_batch_size);

  ~EmbeddingLayer();

 private:
  uint32_t _num_embedding_lookups, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size, _embedding_block_size, _batch_size, _seed;

  float* _embedding_block;
  float* _embedding_gradient;
  float* _embedding_mom;
  float* _embedding_vel;

  float** _embeddings;
  float** _errors;

  uint32_t* _lens;
  uint32_t** _embedding_locs;
};

}  // namespace thirdai::bolt