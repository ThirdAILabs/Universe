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

  void forward(uint32_t batch_indx, const uint32_t* tokens, uint32_t len,
               VectorState& output);

  void backpropagate(uint32_t batch_indx, float learning_rate,
                     const VectorState& output);

  uint32_t getEmbeddingDim() const { return _total_embedding_dim; }

  void initializeLayer(uint32_t new_batch_size);

  BatchState createBatchState(const uint32_t batch_size) {
    return BatchState(_total_embedding_dim, batch_size, true);
  }

  EmbeddingLayer(EmbeddingLayer&) = delete;
  EmbeddingLayer(const EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&&) = delete;

  ~EmbeddingLayer();

 private:
  uint32_t _num_embedding_lookups, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size, _embedding_block_size, _batch_size, _seed;

  float* _embedding_block;

  uint32_t* _loc_lens;
  uint32_t** _embedding_locs;
};

}  // namespace thirdai::bolt