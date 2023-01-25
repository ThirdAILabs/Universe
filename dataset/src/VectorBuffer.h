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
   * Inserts a corresponding vector of BoltVectors (one for each BoltVector
   * stream this buffer is tracking, i.e. the first BoltVector in the vector is
   * from the "first" stream, the second from the "second", and so on). If
   * shuffle is true, each new vector is shuffled with other vectors in its
   * corresponding buffer.
   */
  void insert(std::vector<BoltVector>&& vectors, bool shuffle);

  /**
   * Pops a vector of corresponding batches (one from each BoltVector stream
   * this buffer is tracking, i.e. the first batch in the vector is from the
   * "first" stream, the second from the "second", and so on). Each will be the
   * same size (target_batch_size unless the buffer is low on vectors, which is
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
  void initializeBuffersIfNeeded(const std::vector<BoltVector>& vectors);

  /**
   * Helper method that shuffles the last vector in each buffer with a random
   * other vector in that buffer. If this method is called
   * every time new vectors are added to the buffers, the buffers should be
   * completely shuffled.
   */
  void shuffleNewVectors();

  std::mt19937 _gen;
  /**
   * This data structure consists of a deque storing the currently buffered
   * BoltVectors for every BoltVector stream this VectorBuffer is tracking.
   * Besides during calls to insertBatch or popBatch, this data structure
   * maintains the invariant that BoltVectors with the same index across the
   * buffers are "corresponding"; that is, they were inserted together in a call
   * to insert (additionally, it also maintains the invariant that each deque
   * contains the same number of vectors).
   */
  std::vector<std::deque<BoltVector>> _buffers;
};

}  // namespace thirdai::dataset