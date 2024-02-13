#pragma once

#include <cereal/access.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "LayerConfig.h"
#include <bolt/src/nn/optimizers/Adam.h>
#include <bolt/src/nn/optimizers/Optimizer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/UniversalHash.h>
#include <archive/src/Archive.h>
#include <utils/Random.h>
#include <cmath>
#include <ctime>
#include <optional>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class RobeZ;

namespace tests {
class EmbeddingLayerTestFixture;
}  // namespace tests

class EmbeddingLayer {
  friend class tests::EmbeddingLayerTestFixture;
  friend class RobeZ;

 public:
  explicit EmbeddingLayer(const EmbeddingLayerConfig& config,
                          uint32_t seed = global_random::nextSeed());

  explicit EmbeddingLayer(const ar::Archive& archive);

  void forward(const BoltVector& tokens, BoltVector& output);

  void backpropagate(const BoltVector& tokens, const BoltVector& output);

  void updateParameters(float lr, size_t train_steps);

  uint64_t getOutputDim() const { return _total_embedding_dim; }

  BoltBatch createBatchState(const uint32_t batch_size) const {
    return BoltBatch(_total_embedding_dim, batch_size, true);
  }

  void buildLayerSummary(std::ostream& summary) const;

  void initOptimizer(const OptimizerFactoryPtr& optimizer_factory) {
    // The optimizer may be saved (to preserve state in optimizers like Adam)
    // but the gradients are never saved. Thus we only initialize the optimizer
    // if it's not present, but always initialize the gradients, in case we are
    // initializing the optimizer for a loaded model.

    if (!_optimizer) {
      _optimizer = optimizer_factory->makeOptimizer(
          _embedding_chunks_used.size(), _update_chunk_size);
    }

    _gradients.assign(_embedding_block->size(), 0.0);
  }

  void disableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = true;
  }

  void enableSparseParameterUpdates() {
    _disable_sparse_parameter_updates = false;
  };

  std::vector<float>& getRawEmbeddingBlock() { return *_embedding_block; }

  void saveWithOptimizer(bool should_save_optimizer) {
    _should_serialize_optimizer = should_save_optimizer;
  }

  std::vector<float>& getRawEmbeddingBlockGradient() { return _gradients; }

  std::unique_ptr<EmbeddingLayer> duplicateWithNewReduction(
      const std::string& reduction,
      std::optional<uint64_t> num_tokens_per_input) const;

  uint64_t numEmbeddingLookups() const { return _num_lookups_per_token; }

  uint64_t lookupSize() const { return _lookup_size; }

  uint64_t logEmbeddingBlockSize() const { return _log_embedding_block_size; }

  std::string reduction() const {
    switch (_reduction) {
      case EmbeddingReductionType::AVERAGE:
        return "avg";
      case EmbeddingReductionType::SUM:
        return "sum";
      case EmbeddingReductionType::CONCATENATION:
        return "concat";
      default:
        return "";
    }
  }

  std::optional<uint64_t> numTokensPerInput() const {
    return _num_tokens_per_input;
  }

  uint64_t updateChunkSize() const { return _update_chunk_size; }

  uint32_t hashSeed() const { return _hash_fn.seed(); }

  bool hasOptimizer() const { return _optimizer != nullptr; }

  ~EmbeddingLayer() = default;

 private:
  inline uint64_t getEmbeddingBlockOffset(uint32_t token,
                                          uint64_t lookup_index) {
    uint64_t id = token * _num_lookups_per_token + lookup_index;
    uint64_t hash = _hash_fn.gethash(id);

    // We bit shift to make sure that the hash loc is within the range of the
    // embedding block.
    return hash >> (64 - _log_embedding_block_size);
  }

  constexpr uint64_t getOutputOffsetWithinEmbedding(
      uint64_t lookup_index) const {
    return lookup_index * _lookup_size;
  }

  void markUsedChunks(uint64_t block_offset) {
    for (uint64_t i = block_offset; i < block_offset + _lookup_size;
         i += _update_chunk_size) {
      _embedding_chunks_used[i / _update_chunk_size] = true;
    }
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
            _log_embedding_block_size, _update_chunk_size, _reduction,
            _num_tokens_per_input, _embedding_block_size, _hash_fn,
            _embedding_block, _embedding_chunks_used,
            _disable_sparse_parameter_updates, _should_serialize_optimizer);

    if (_should_serialize_optimizer &&
        std::is_same_v<Archive, cereal::BinaryInputArchive>) {
      AdamOptimizer optimizer;

      archive(optimizer);

      _optimizer = Adam::fromOldOptimizer(std::move(optimizer),
                                          _embedding_chunks_used.size(),
                                          _update_chunk_size);

      _gradients.assign(_embedding_block->size(), 0.0);
    }
  }

  uint64_t _num_lookups_per_token;
  uint64_t _lookup_size;
  uint64_t _total_embedding_dim;
  uint64_t _log_embedding_block_size;
  uint64_t _update_chunk_size;
  uint64_t _embedding_block_size;

  EmbeddingReductionType _reduction;
  std::optional<uint64_t> _num_tokens_per_input;

  hashing::UniversalHash _hash_fn;

  std::shared_ptr<std::vector<float>> _embedding_block;

  /**
   * The embedding block is grouped into chunks of size _update_chunk_size.
   * During backpropagation the layer tracks which chunks are used and gradients
   * are computed for. Then during update parameters only the used chunks are
   * updated.
   */
  std::vector<bool> _embedding_chunks_used;

  std::vector<float> _gradients;
  OptimizerPtr _optimizer;
  bool _should_serialize_optimizer;

  bool _disable_sparse_parameter_updates;
};

}  // namespace thirdai::bolt