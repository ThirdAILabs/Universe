#pragma once

#include "Layer.h"
#include "LayerConfig.h"
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

  void forward(uint32_t batch_indx, const std::vector<uint32_t>& tokens,
               VectorState& output);

  void backpropagate(uint32_t batch_indx, const VectorState& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  uint32_t getEmbeddingDim() const { return _total_embedding_dim; }

  void initializeLayer(uint32_t new_batch_size);

  BatchState createBatchState(const uint32_t batch_size) const {
    return BatchState(_total_embedding_dim, batch_size, true);
  }

  EmbeddingLayer(EmbeddingLayer&) = delete;
  EmbeddingLayer(const EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&&) = delete;

  ~EmbeddingLayer();

 private:
  std::vector<std::pair<uint64_t, uint64_t>> getDisjointUpdateRanges();

  uint32_t _num_embedding_lookups, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size, _embedding_block_size, _batch_size, _seed;

  float* _embedding_block;
  float* _gradients;
  float* _momentum;
  float* _velocity;

  uint32_t* _loc_lens;
  uint32_t** _embedding_locs;
};

}  // namespace thirdai::bolt