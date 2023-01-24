#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <random>

namespace thirdai::dataset {

/**
 * This class manages a buffer for N different corresponding BoltVector streams.
 * When adding to the buffers, it takes in N lists that each have M BoltVectors,
 * where M is  arbitrary. When removing from the buffer, it returns N batches
 * at a time that each have K vectors, where K is specified by the user.
 * Optionally, this class will also shuffle vectors in the buffers (while
 * keeping each corresponding set of vectors the same),
 */
class VectorBuffer {
 public:
  explicit VectorBuffer(uint32_t shuffle_seed) : _gen(shuffle_seed) {}

  /**
   * Inserts a corresponding vector of batches (one from each dataset this
   * buffer is tracking, i.e. the first batch in the vector is from the "first"
   * dataset, the second from the "second", and so on). If shuffle is true,
   * each new vector is shuffled with other vectors in the buffer.
   */
  void insertBatch(std::vector<BoltBatch>&& batches, bool shuffle);

  /**
   * Pops a vector of batches (one from each dataset this
   * buffer is tracking, i.e. the first batch in the vector is from the "first"
   * dataset, the second from the "second", and so on). Each will be the same
   * size (target_batch_size unless the buffer is low on vectors, which is
   * usually because the source has been exhausted).
   */
  std::optional<std::vector<BoltBatch>> popBatch(size_t target_batch_size);

  /**
   * Similar to popBatch, except the first element is up to num_batches from
   * the first dataset, the second element is up to num_batches from the
   * second dataset, and so on. All vectors of batches will have the same size,
   * and each corresponding batch size will be the same (all target_batch_size
   * except possible the last one).
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

  /**
   * Shuffles only the vectors we have just added in the passed in buffers
   * by swapping them with other vectors in the buffer. If this method is called
   * every time new vectors are added to the buffer, the buffer should be
   * completely shuffled.
   */
  static inline void swapShuffle(std::vector<std::deque<BoltVector>>& buffers,
                                 size_t batch_size_added, std::mt19937& gen);

  std::mt19937 _gen;
  /**
   * This data structure consists of a deque for every dataset this VectorBuffer
   * is tracking. Thus, besides during calls to insertBatch or popBatch, this
   * data structure thus maintains the invariant that every deque contains the
   * same number of vectors.
   */
  std::vector<std::deque<BoltVector>> _buffers;
};

}  // namespace thirdai::dataset