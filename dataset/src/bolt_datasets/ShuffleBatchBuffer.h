#include <bolt/src/layers/BoltVector.h>
#include <cstddef>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::dataset {

class ShuffleBatchBuffer {
 public:
  ShuffleBatchBuffer(size_t n_batches_in_buffer, size_t batch_size,
                     uint32_t shuffle_seed)
      : _rand(shuffle_seed),
        _last_batch(false),
        _batch_size(batch_size),
        _insert_batch_idx(0),
        _pop_batch_idx(0),
        _current_n_batches(0),
        _batches(n_batches_in_buffer) {}

  void insertBatch(std::pair<bolt::BoltBatch, bolt::BoltBatch>&& batch, bool resize_if_full) {
    if (_last_batch) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch after "
          "last batch (batch with smaller number of vectors).");
    }

    if (batch.first.getBatchSize() > _batch_size) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "batch with more vectors than expected (expected = "
               << _batch_size << " actual = " << batch.first.getBatchSize()
               << ").";
      throw std::runtime_error(error_ss.str());
    }

    bool full = _current_n_batches == _batches.size();

    if (full && !resize_if_full) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch to a "
          "full buffer with resize_if_full = False.");
    }
    
    if (full && _pop_batch_idx != 0) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to resize "
          "non-contiguous buffer. ShuffleBatchBuffer uses a circular "
          "buffer and cannot be resized after popping a batch.");
    }

    if (full) {
      _insert_batch_idx = _batches.size();
      _batches.emplace_back();
    }
    _batches[_insert_batch_idx] = std::move(batch);
    handleShuffle(/* inserted_batch = */ _batches[_insert_batch_idx],
                  /* n_batches_before_insertion = */ _current_n_batches);

    if (_batches[_insert_batch_idx].first.getBatchSize() < _batch_size) {
      _last_batch = true;
    }

    _insert_batch_idx = (_insert_batch_idx + 1) % _batches.size();
    _current_n_batches++;
  }

  std::optional<std::pair<bolt::BoltBatch, bolt::BoltBatch>> popBatch() {
    if (_current_n_batches == 0) {
      return {};
    }
    auto popped = std::move(_batches[_pop_batch_idx]);

    _pop_batch_idx = (_pop_batch_idx + 1) % _batches.size();
    _current_n_batches--;

    return popped;
  }

 private:
  inline void handleShuffle(
      std::pair<bolt::BoltBatch, bolt::BoltBatch>& inserted_batch,
      size_t n_batches_before_insertion) {
    for (size_t i = 0; i < inserted_batch.first.getBatchSize(); i++) {
      size_t swap_with = _rand();
      if (swap_with < n_batches_before_insertion) {
        size_t swap_batch_pos = swap_with / _batch_size;
        size_t swap_batch_idx =
            (_pop_batch_idx + swap_batch_pos) % _batches.size();
        size_t swap_vec_idx = swap_with % _batch_size;
        auto& swap_batch = _batches[swap_batch_idx];
        swapVecs(swap_batch.first[swap_vec_idx], inserted_batch.first[i]);
        swapVecs(swap_batch.second[swap_vec_idx], inserted_batch.second[i]);
      }
    }
  }

  static inline void swapVecs(bolt::BoltVector& first,
                              bolt::BoltVector& second) {
    bolt::BoltVector temp = std::move(first);
    first = std::move(second);
    second = std::move(temp);
  }

  std::mt19937 _rand;

  bool _last_batch;
  size_t _batch_size;

  size_t _insert_batch_idx;
  size_t _pop_batch_idx;
  size_t _current_n_batches;
  std::vector<std::pair<bolt::BoltBatch, bolt::BoltBatch>> _batches;
};

}  // namespace thirdai::dataset