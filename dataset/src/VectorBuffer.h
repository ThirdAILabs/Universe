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
  explicit VectorBuffer(bool should_shuffle, uint32_t shuffle_seed,
                        size_t num_datasets)
      : _shuffle(should_shuffle), _gen(shuffle_seed), _buffers(num_datasets) {}

  /**
   * Inserts a corresponding vector of BoltVectors (one for each BoltVector
   * stream this buffer is tracking, i.e. the first BoltVector in the vector is
   * from the "first" stream, the second from the "second", and so on).
   */
  void insert(std::vector<BoltVector>&& vectors);

  /**
   * Pops a vector of corresponding of BoltVectors (one for each BoltVector
   * stream this buffer is tracking, i.e. the first BoltVector in the vector is
   * from the "first" stream, the second from the "second", and so on). Returns
   * std::nullopt if the streams are empty.
   */
  std::optional<std::vector<BoltVector>> pop();

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

  // Whether we should shuffle the vectors that get inserted with other vectors
  // in the buffer
  bool _shuffle;
  // Random number generator to use for shuffling
  std::mt19937 _gen;

  /**
   * This data structure consists of a deque storing the currently buffered
   * BoltVectors for every BoltVector stream this VectorBuffer is tracking.
   * Besides during calls to insert or popBatch, this data structure
   * maintains the invariant that BoltVectors with the same index across the
   * buffers are "corresponding"; that is, they were inserted together in a call
   * to insert (additionally, it also maintains the invariant that each deque
   * contains the same number of vectors).
   */
  std::vector<std::deque<BoltVector>> _buffers;
};

}  // namespace thirdai::dataset