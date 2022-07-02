#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/utils/CircularBuffer.h>
#include <ctime>
#include <optional>
#include <random>
#include <sstream>

namespace thirdai::dataset {

class ShuffleBatchBuffer {
 public:
  explicit ShuffleBatchBuffer(uint32_t shuffle_seed)
      : _gen(shuffle_seed), _saw_last_batch(false), _batch_size(0) {}

  void insertBatch(std::pair<bolt::BoltBatch, bolt::BoltBatch>&& batch,
                   bool shuffle) {
    if (empty()) {
      _batch_size = batch.first.getBatchSize();
      _saw_last_batch = false;
    }

    checkConsistentBatchSize(batch.first.getBatchSize());

    if (shuffle) {
      swapShuffle(_input_batches, batch.first, _label_batches, batch.second,
                  _batch_size, _gen);
    }

    _input_batches.insert(std::move(batch.first));
    _label_batches.insert(std::move(batch.second));
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

    auto len = input_batches.size() * _batch_size;
    if (!input_batches.empty()) {
      len = len - _batch_size + input_batches.back().getBatchSize();
    }

    return {std::make_shared<BoltDataset>(std::move(input_batches), len),
            std::make_shared<BoltDataset>(std::move(label_batches), len)};
  }

  inline bool empty() const { return _input_batches.empty(); }

 private:
  inline void checkConsistentBatchSize(size_t batch_size) {
    if (_saw_last_batch) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch after "
          "last batch (batch with smaller number of vectors).");
    }

    if (batch_size > _batch_size) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "batch with more vectors than expected (expected = "
               << _batch_size << " actual = " << batch_size << ").";
      throw std::runtime_error(error_ss.str());
    }

    if (batch_size < _batch_size) {
      _saw_last_batch = true;
    }
  }

  static inline void swapShuffle(CircularBuffer<bolt::BoltBatch>& input_batches,
                                 bolt::BoltBatch& new_input_batch,
                                 CircularBuffer<bolt::BoltBatch>& label_batches,
                                 bolt::BoltBatch& new_label_batch,
                                 size_t expected_batch_size,
                                 std::mt19937& gen) {
    for (size_t i = 0; i < new_input_batch.getBatchSize(); i++) {
      size_t rand_range = input_batches.size() + new_input_batch.getBatchSize();
      std::uniform_int_distribution<> dist(0, rand_range);
      size_t swap_with = dist(gen);
      if (swap_with < input_batches.size()) {
        size_t swap_batch_pos = swap_with / expected_batch_size;
        size_t swap_vec_idx = swap_with % expected_batch_size;

        swapVecs(input_batches.at(swap_batch_pos)[swap_vec_idx],
                 new_input_batch[i]);
        swapVecs(label_batches.at(swap_batch_pos)[swap_vec_idx],
                 new_label_batch[i]);
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

  CircularBuffer<bolt::BoltBatch> _input_batches;
  CircularBuffer<bolt::BoltBatch> _label_batches;
};

struct ShuffleBufferConfig {
  ShuffleBufferConfig() : buffer_size(1000), seed(time(NULL)) {}
  explicit ShuffleBufferConfig(size_t buffer_size)
      : buffer_size(buffer_size), seed(time(NULL)) {}
  ShuffleBufferConfig(size_t buffer_size, uint32_t seed)
      : buffer_size(buffer_size), seed(seed) {}

  size_t buffer_size;
  uint32_t seed;
};

}  // namespace thirdai::dataset