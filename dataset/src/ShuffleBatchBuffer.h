#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <ctime>
#include <deque>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>

namespace thirdai::dataset {

class ShuffleBatchBuffer {
 public:
  explicit ShuffleBatchBuffer(uint32_t shuffle_seed, size_t batch_size)
      : _gen(shuffle_seed),
        _reached_end_of_dataset(false),
        _batch_size(batch_size) {}

  void insertBatch(std::vector<BoltBatch>&& batches, bool shuffle) {
    checkConsistentBatchSize(batches);

    initializeBuffersIfNeeded(batches);

    for (uint32_t i = 0; i < batches.size(); i++) {
      _batch_buffers.at(i).push_back(std::move(batches.at(i)));
    }

    if (shuffle) {
      swapShuffle(_batch_buffers, _batch_size, _gen);
    }
  }

  std::optional<std::vector<BoltBatch>> popBatch() {
    if (empty()) {
      return std::nullopt;
    }

    std::vector<BoltBatch> batches;
    for (auto& batch_buffer : _batch_buffers) {
      batches.push_back(std::move(batch_buffer.front()));
      batch_buffer.pop_front();
    }

    return batches;
  }

  /**
   * Exports min(num_batches, size()) batches into a vector of vector of
   * BoltBatch
   */
  std::vector<std::vector<BoltBatch>> exportBuffer(uint64_t num_batches) {
    uint64_t num_batches_in_result = std::min<uint64_t>(num_batches, size());
    /*
      This doesn't double our memory footprint since the
      batches are moved;
      the amount of memory allocated to the underlying
      vectors remains the same.
    */
    std::vector<std::vector<BoltBatch>> exported_batch_lists;
    for (auto& batch_buffer : _batch_buffers) {
      std::vector<BoltBatch> batch_list(
          std::make_move_iterator(batch_buffer.begin()),
          std::make_move_iterator(batch_buffer.begin() +
                                  num_batches_in_result));
      exported_batch_lists.push_back(std::move(batch_list));
      for (uint32_t i = 0; i < num_batches_in_result; i++) {
        batch_buffer.pop_front();
      }
    }

    return exported_batch_lists;
  }

  inline bool empty() const {
    return _batch_buffers.empty() || _batch_buffers.at(0).empty();
  }

  size_t size() const {
    if (empty()) {
      return 0;
    }
    return _batch_buffers.at(0).size();
  }

  void clear() {
    _batch_buffers.clear();
    _reached_end_of_dataset = false;
  }

 private:
  void initializeBuffersIfNeeded(const std::vector<BoltBatch>& batches) {
    if (_batch_buffers.empty()) {
      _batch_buffers = std::vector<std::deque<BoltBatch>>(batches.size());
    }

    if (_batch_buffers.size() != batches.size()) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "a different number of corresponding batches than originally "
                  "inserted into the buffer (originally inserted "
               << _batch_buffers.size() << ", trying to insert "
               << batches.size() << ").";
      throw std::runtime_error(error_ss.str());
    }
  }

  inline void checkConsistentBatchSize(const std::vector<BoltBatch>& batches) {
    if (batches.empty()) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Expected at least one "
          "batch to be inserted for shuffling but found 0.");
    }
    uint32_t first_data_batch_size = batches.at(0).getBatchSize();
    for (uint32_t i = 1; i < batches.size(); i++) {
      if (batches.at(i).getBatchSize() != first_data_batch_size) {
        std::stringstream error_ss;
        error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                    "corresponding batches with different sizes (one size = "
                 << first_data_batch_size
                 << ", the other size = " << batches.at(i).getBatchSize()
                 << ").";
        throw std::runtime_error(error_ss.str());
      }
    }

    if (_reached_end_of_dataset) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch after "
          "reaching the end of the dataset.");
    }

    if (first_data_batch_size > _batch_size) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "batch that is larger than expected (expected size = "
               << _batch_size << " actual = " << first_data_batch_size << ").";
      throw std::runtime_error(error_ss.str());
    }

    if (first_data_batch_size < _batch_size) {
      _reached_end_of_dataset = true;
    }
  }

  static inline void swapShuffle(
      std::vector<std::deque<BoltBatch>>& batch_lists,
      size_t expected_batch_size, std::mt19937& gen) {
    assert(batch_lists.at(0).size() > 0);
    size_t n_old_vecs = (batch_lists.at(0).size() - 1) * expected_batch_size;
    size_t n_vecs = n_old_vecs + batch_lists.at(0).back().getBatchSize();
    std::uniform_int_distribution<> dist(
        0, n_vecs - 1);  // Accepts a closed interval

    for (size_t i = 0; i < batch_lists.at(0).back().getBatchSize(); i++) {
      size_t swap_with = dist(gen);
      /*
        Only swap with vectors in old batches for two reasons:
        1. Swapping with elements in the same batch is effectively a no-op
           since vectors in the same batch are processed by bolt in parallel
        2. This ensures that each element in the new batch to has an equal
           probability of being swapped out of this batch.
      */
      if (swap_with < n_old_vecs) {
        size_t swap_batch_idx = swap_with / expected_batch_size;
        size_t swap_vec_idx = swap_with % expected_batch_size;

        for (auto& batch_list : batch_lists) {
          std::swap(batch_list.at(swap_batch_idx)[swap_vec_idx],
                    batch_list.back()[i]);
        }
      }
    }
  }

  std::mt19937 _gen;

  bool _reached_end_of_dataset;
  size_t _batch_size;

  std::vector<std::deque<BoltBatch>> _batch_buffers;
};

}  // namespace thirdai::dataset