#pragma once

#include "Layer.h"
#include "LayerConfig.h"
#include <ctime>

namespace thirdai::bolt {

namespace tests {
class EmbeddingLayerTestFixture;
}  // namespace tests

class EmbeddingLayer {
  friend class tests::EmbeddingLayerTestFixture;

 public:
  explicit EmbeddingLayer(const EmbeddingLayerConfig& config,
                          uint32_t seed = time(nullptr));

  void feedForward(uint32_t batch_indx, const uint32_t* tokens, uint32_t len);

  void backpropagate(uint32_t batch_indx, float learning_rate);

  uint32_t getEmbeddingDim() const { return _total_embedding_dim; }

  const float* getEmbedding(uint32_t batch_indx) const {
    return _embeddings[batch_indx];
  }

  float* getErrors(uint32_t batch_indx) { return _errors[batch_indx]; }

  void initializeLayer(uint32_t new_batch_size);

  // This should not be used with setBatchSize
  void initializeLayer(uint32_t batch_size, float** new_embeddings,
                       float** new_errors);

  EmbeddingLayer(EmbeddingLayer&) = delete;
  EmbeddingLayer(const EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&&) = delete;

  ~EmbeddingLayer();

 private:
  void deallocateInternalState();

  uint32_t _num_embedding_lookups, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size, _embedding_block_size, _batch_size, _seed;

  float* _embedding_block;

  float** _embeddings;
  float** _errors;

  bool _internal_state_provided;

  uint32_t* _loc_lens;
  uint32_t** _embedding_locs;
};

}  // namespace thirdai::bolt