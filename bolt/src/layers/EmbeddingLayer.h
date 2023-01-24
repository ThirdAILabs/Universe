#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include <bolt/src/layers/Optimizer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/UniversalHash.h>
#include <cmath>
#include <ctime>
#include <optional>
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

  void forward(uint32_t vec_index, const BoltVector& tokens,
               BoltVector& output);

  void backpropagate(uint32_t vec_index, const BoltVector& output);

  void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

  uint32_t getOutputDim() const { return _total_embedding_dim; }

  void initializeLayer(uint32_t new_batch_size);

  BoltBatch createBatchState(const uint32_t batch_size) const {
    return BoltBatch(_total_embedding_dim, batch_size, true);
  }

  void buildLayerSummary(std::stringstream& summary) const;

  void initOptimizer() {
    if (!_optimizer) {
      _optimizer = AdamOptimizer(_embedding_block_size);
    }
  }

  void disableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = true;
  }

  void saveWithOptimizer(bool should_save_optimizer) {
    _should_save_optimizer = should_save_optimizer;
  }

  std::vector<float>& getRawEmbeddingBlock() { return _embedding_block; }

  std::vector<float>& getRawEmbeddingBlockGradient() {
    return _optimizer->gradients;
  }

  EmbeddingLayer(const EmbeddingLayer&) = delete;
  EmbeddingLayer(EmbeddingLayer&&) = delete;
  EmbeddingLayer& operator=(const EmbeddingLayer&) = delete;
  EmbeddingLayer& operator=(EmbeddingLayer&&) = delete;

  ~EmbeddingLayer() = default;

 private:
  std::vector<std::pair<uint64_t, uint64_t>> getDisjointUpdateRanges() const;

  void updateParametersSparse(float lr, uint32_t iter, float B1, float B2,
                              float eps);

  void updateParametersDense(float lr, uint32_t iter, float B1, float B2,
                             float eps);

  inline uint32_t getEmbeddingBlockOffset(uint32_t token,
                                          uint32_t lookup_index) {
    uint64_t id = token * _num_lookups_per_token + lookup_index;
    uint32_t hash = _hash_fn.gethash(id);

    // We bit shift to make sure that the hash loc is within the range of the
    // embedding block.
    return hash >> (32 - _log_embedding_block_size);
  }

  constexpr uint32_t getOutputOffsetWithinEmbedding(
      uint32_t lookup_index) const {
    return lookup_index * _lookup_size;
  }

  static constexpr uint32_t getHashLocIndex(uint32_t lookup_index,
                                            uint32_t token_index,
                                            uint32_t num_tokens) {
    return lookup_index * num_tokens + token_index;
  }

  void recordEmbeddingBlockOffset(uint32_t vec_index, uint64_t hash_loc) {
    _embedding_block_offsets[vec_index].push_back(hash_loc);
  }

  uint64_t retrieveEmbeddingBlockOffset(uint32_t vec_index,
                                        uint32_t lookup_index,
                                        uint32_t token_index,
                                        uint32_t num_tokens) {
    return _embedding_block_offsets[vec_index][getHashLocIndex(
        lookup_index, token_index, num_tokens)];
  }

  // Private constructor for cereal.
  EmbeddingLayer() : _hash_fn(0) {}

  /**
   * Training data-structures (like the optimizer and the active neurons
   * trackers) are not loaded in by default. If we want to continue training
   * after a load, the expectation is that the higher level Graph/Network API
   * will handle this initialization with the initOptimizer() method.
   *
   * Doing this means our load API is as simple as possible for both
   * training and inference purposes. It doesn't make sense to load these
   * data-structures by default then remove them with another function since
   * users may be memory constrained during deployment.
   *
   * We don't know yet if its worth it to save the optimizer for
   * retraining/finetuning purposes. If in the future we figure out this has
   * some benefit we can adjust this method accordingly.
   */
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_lookups_per_token, _lookup_size, _total_embedding_dim,
            _log_embedding_block_size, _reduction, _num_tokens_per_input,
            _embedding_block_size, _hash_fn, _embedding_block,
            _disable_sparse_parameter_updates, _should_save_optimizer);
    if (_should_save_optimizer) {
      archive(_optimizer);
    }
  }

  uint32_t _num_lookups_per_token, _lookup_size, _total_embedding_dim,
      _log_embedding_block_size;
  uint64_t _embedding_block_size;
  EmbeddingReductionType _reduction;
  std::optional<uint32_t> _num_tokens_per_input;

  hashing::UniversalHash _hash_fn;

  std::vector<float> _embedding_block;

  std::optional<AdamOptimizer> _optimizer = std::nullopt;
  bool _disable_sparse_parameter_updates;

  // A flag to determine whether the current network saves the optimizer states
  // or not. If true, it saves the optimizer states, else doesn't.
  bool _should_save_optimizer;

  // This structure stores the embedding block offset for each token in each
  // input. This is used for backpropagation and for update paramters to know
  // what parts of the embedding block were used.
  std::vector<std::vector<uint64_t>> _embedding_block_offsets;
};

}  // namespace thirdai::bolt