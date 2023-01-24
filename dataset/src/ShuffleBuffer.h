#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <random>

namespace thirdai::dataset {

class ShuffleBuffer {
 public:
  explicit ShuffleBuffer(uint32_t shuffle_seed) : _gen(shuffle_seed) {}

  void insertBatch(std::vector<BoltBatch>&& batches, bool shuffle);

  std::optional<std::vector<BoltBatch>> popBatch(size_t target_batch_size);

  /**
   * Pops min(num_batches, size()) batches into a vector of vector of
   * BoltBatch (with batch size target_batch_size, except possible the last
   * entry in each vector which may be smaller)
   */
  std::vector<std::vector<BoltBatch>> popBatches(size_t num_batches,
                                                 size_t target_batch_size);

  inline bool empty() const {
    return _buffers.empty() || _buffers.at(0).empty();
  }

  size_t size() const {
    if (empty()) {
      return 0;
    }
    return _buffers.at(0).size();
  }

  void clear() { _buffers.clear(); }

 private:
  void initializeBuffersIfNeeded(const std::vector<BoltBatch>& batches);

  static inline void checkConsistentBatchSize(
      const std::vector<BoltBatch>& batches);

  static inline void swapShuffle(std::vector<std::deque<BoltVector>>& buffers,
                                 size_t batch_size_added, std::mt19937& gen);

  std::mt19937 _gen;
  /**
   * Besides during calls to addBatch or popBatch, this data structure thus
   * maintains the invariant that every deque contains the same number of
   * vectors.
   */
  std::vector<std::deque<BoltVector>> _buffers;
};

}  // namespace thirdai::dataset