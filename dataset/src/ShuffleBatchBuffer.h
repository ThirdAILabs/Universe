#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/utils/CircularQueue.h>
#include <ctime>
#include <optional>
#include <random>
#include <sstream>

namespace thirdai::dataset {

class ShuffleBatchBuffer {
 public:
  explicit ShuffleBatchBuffer(uint32_t shuffle_seed, size_t batch_size)
      : _gen(shuffle_seed), _saw_last_batch(false), _batch_size(batch_size) {}

  void insertBatch(std::tuple<bolt::BoltBatch, bolt::BoltBatch>&& batch,
                   bool shuffle) {
    checkConsistentBatchSize(std::get<0>(batch).getBatchSize());

    _input_batches.insert(std::move(std::get<0>(batch)));
    _label_batches.insert(std::move(std::get<1>(batch)));

    if (shuffle) {
      swapShuffle(_input_batches, _label_batches, _batch_size, _gen);
    }
  }

  std::optional<std::pair<bolt::BoltBatch, bolt::BoltBatch>> popBatch() {
    auto input_batch = _input_batches.pop();
    auto label_batch = _label_batches.pop();
    if (!input_batch) {
      return {};
    }
    return {{std::move(input_batch.value()), std::move(label_batch.value())}};
  }

  std::pair<BoltDatasetPtr, BoltDatasetPtr> exportBuffer() {
    auto input_batches = _input_batches.exportContiguousBuffer();
    auto label_batches = _label_batches.exportContiguousBuffer();

    return {std::make_shared<BoltDataset>(std::move(input_batches)),
            std::make_shared<BoltDataset>(std::move(label_batches))};
  }

  inline bool empty() const { return _input_batches.empty(); }

 private:
  inline void checkConsistentBatchSize(size_t new_batch_size) {
    if (_saw_last_batch) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch after "
          "last batch (batch with smaller size than expected is treated as "
          "the last batch).");
    }

    if (new_batch_size > _batch_size) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "batch that is larger than expected (expected size = "
               << _batch_size << " actual = " << new_batch_size << ").";
      throw std::runtime_error(error_ss.str());
    }

    if (new_batch_size < _batch_size) {
      _saw_last_batch = true;
    }
  }

  static inline void swapShuffle(CircularQueue<bolt::BoltBatch>& input_batches,
                                 CircularQueue<bolt::BoltBatch>& label_batches,
                                 size_t expected_batch_size,
                                 std::mt19937& gen) {
    assert(input_batches.size() > 0);
    size_t n_old_vecs = (input_batches.size() - 1) * expected_batch_size;
    size_t n_vecs = n_old_vecs + input_batches.last().getBatchSize();
    std::uniform_int_distribution<> dist(0, n_vecs);

    for (size_t i = 0; i < input_batches.last().getBatchSize(); i++) {
      size_t swap_with = dist(gen);
      /*
        Only swap with vectors in old batches for two reasons:
        1. Swapping with elements in the same batch is effectively a no-op
           since vectors in the same batch are processed by bolt in parallel
        2. This ensures that each element in the new batch to has an equal
           probability of being swapped out of this batch.
      */
      if (swap_with < n_old_vecs) {
        size_t swap_batch_pos = swap_with / expected_batch_size;
        size_t swap_vec_idx = swap_with % expected_batch_size;

        swapVecs(input_batches.at(swap_batch_pos)[swap_vec_idx],
                 input_batches.last()[i]);
        swapVecs(label_batches.at(swap_batch_pos)[swap_vec_idx],
                 label_batches.last()[i]);
      }
    }
  }

  static inline void swapVecs(bolt::BoltVector& first,
                              bolt::BoltVector& second) {
    bolt::BoltVector temp = std::move(first);
    first = std::move(second);
    second = std::move(temp);
  }

  std::mt19937 _gen;

  bool _saw_last_batch;
  size_t _batch_size;

  CircularQueue<bolt::BoltBatch> _input_batches;
  CircularQueue<bolt::BoltBatch> _label_batches;
};

struct ShuffleBufferConfig {
  ShuffleBufferConfig() : n_batches(1000), seed(time(NULL)) {}

  explicit ShuffleBufferConfig(size_t buffer_size)
      : n_batches(buffer_size), seed(time(NULL)) {}

  ShuffleBufferConfig(size_t buffer_size, uint32_t seed)
      : n_batches(buffer_size), seed(seed) {}

  size_t n_batches;
  uint32_t seed;
};

}  // namespace thirdai::dataset