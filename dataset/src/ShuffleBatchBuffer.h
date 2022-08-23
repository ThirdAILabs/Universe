#pragma once

#include <bolt/src/layers/BoltVector.h>
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

  void insertBatch(std::tuple<bolt::BoltBatch, bolt::BoltBatch>&& batch,
                   bool shuffle) {
    checkConsistentBatchSize(std::get<0>(batch).getBatchSize(),
                             std::get<1>(batch).getBatchSize());

    _input_batches.push_back(std::move(std::get<0>(batch)));
    _label_batches.push_back(std::move(std::get<1>(batch)));

    if (shuffle) {
      swapShuffle(_input_batches, _label_batches, _batch_size, _gen);
    }
  }

  std::optional<std::pair<bolt::BoltBatch, bolt::BoltBatch>> popBatch() {
    assert(_input_batches.empty() == _label_batches.empty());
    if (_input_batches.empty()) {
      return {};
    }

    auto input_batch = std::move(_input_batches.front());
    auto label_batch = std::move(_label_batches.front());
    _input_batches.pop_front();
    _label_batches.pop_front();
    return {{std::move(input_batch), std::move(label_batch)}};
  }

  std::pair<std::vector<bolt::BoltBatch>, std::vector<bolt::BoltBatch>>
  exportBuffer() {
    /*
      This doesn't double our memory footprint since the
      batches are moved;
      the amount of memory allocated to the underlying
      vectors remains the same.
    */
    std::vector<bolt::BoltBatch> input_batch_vector(
        std::make_move_iterator(_input_batches.begin()),
        std::make_move_iterator(_input_batches.end()));
    std::vector<bolt::BoltBatch> label_batch_vector(
        std::make_move_iterator(_label_batches.begin()),
        std::make_move_iterator(_label_batches.end()));

    _input_batches.clear();
    _label_batches.clear();

    return {std::move(input_batch_vector), std::move(label_batch_vector)};
  }

  inline bool empty() const { return _input_batches.empty(); }

 private:
  inline void checkConsistentBatchSize(size_t new_input_batch_size,
                                       size_t new_label_batch_size) {
    if (new_input_batch_size != new_label_batch_size) {
      std::stringstream error_ss;
      error_ss
          << "[ShuffleBatchBuffer::insertBatch] Attempted to instert input and "
             "label batches with different sizes (input batch size = "
          << new_input_batch_size
          << ", label batch size = " << new_label_batch_size << ").";
      throw std::runtime_error(error_ss.str());
    }

    if (_reached_end_of_dataset) {
      throw std::runtime_error(
          "[ShuffleBatchBuffer::insertBatch] Attempted to insert batch after "
          "reaching the end of the dataset.");
    }

    if (new_input_batch_size > _batch_size) {
      std::stringstream error_ss;
      error_ss << "[ShuffleBatchBuffer::insertBatch] Attempted to insert "
                  "batch that is larger than expected (expected size = "
               << _batch_size << " actual = " << new_input_batch_size << ").";
      throw std::runtime_error(error_ss.str());
    }

    if (new_input_batch_size < _batch_size) {
      _reached_end_of_dataset = true;
    }
  }

  static inline void swapShuffle(std::deque<bolt::BoltBatch>& input_batches,
                                 std::deque<bolt::BoltBatch>& label_batches,
                                 size_t expected_batch_size,
                                 std::mt19937& gen) {
    assert(input_batches.size() > 0);
    size_t n_old_vecs = (input_batches.size() - 1) * expected_batch_size;
    size_t n_vecs = n_old_vecs + input_batches.back().getBatchSize();
    std::uniform_int_distribution<> dist(
        0, n_vecs - 1);  // Accepts a closed interval

    for (size_t i = 0; i < input_batches.back().getBatchSize(); i++) {
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

        std::swap(input_batches.at(swap_batch_idx)[swap_vec_idx],
                  input_batches.back()[i]);
        std::swap(label_batches.at(swap_batch_idx)[swap_vec_idx],
                  label_batches.back()[i]);
      }
    }
  }

  std::mt19937 _gen;

  bool _reached_end_of_dataset;
  size_t _batch_size;

  std::deque<bolt::BoltBatch> _input_batches;
  std::deque<bolt::BoltBatch> _label_batches;
};

}  // namespace thirdai::dataset