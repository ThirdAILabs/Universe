#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <random>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * This class manages a buffer for num_datasets different corresponding
 * BoltVector streams. When inserting, the user should pass in lists of size
 * num_datasets of corresponding BoltVectors, and when popping the user will
 * also receive lists of size num_datasets of corresponding BoltVectors.
 * The main utility of this class is that it can maintain a shuffled state for
 * all vectors currently inserted (while maintaining the correspondence between
 * vectors); if should_shuffle is false then this shuffling is not done
 * and this class serves as a simple FIFO queue.
 */
class VectorBuffer {
 public:
  explicit VectorBuffer(bool should_shuffle, uint32_t shuffle_seed,
                        size_t num_datasets)
      : _shuffle(should_shuffle), _gen(shuffle_seed), _buffers(num_datasets) {
    if (num_datasets == 0) {
      throw std::invalid_argument(
          "The buffer must be for at least 1 dataset, but was constructed with "
          "a value of 0 for num_datasets.");
    }
  }

  /**
   * Inserts a vector of num_datasets corresponding BoltVectors (one for each
   * BoltVector dataset/stream this buffer is tracking, i.e. the first
   * BoltVector in the vector is from the "first" stream, the second from the
   * "second", and so on).
   */
  void insert(std::vector<BoltVector>&& vectors);

  /**
   * Pops a vector of num_datasets corresponding BoltVectors (one for each
   * BoltVector stream this buffer is tracking, i.e. the first BoltVector in the
   * vector is from the "first" stream, the second from the "second", and so
   * on). Returns std::nullopt if the buffers are empty.
   */
  std::optional<std::vector<BoltVector>> pop();

  inline bool empty() const { return size() == 0; }

  size_t size() const { return _buffers.at(0).size(); }

 private:
  void verifyCorrectNumberOfVectors(const std::vector<BoltVector>& vectors);

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
   * Besides during calls to insert or pop, this data structure
   * maintains the invariant that BoltVectors with the same index across the
   * buffers are "corresponding"; that is, they were inserted together in a call
   * to insert (additionally, it also maintains the invariant that each deque
   * contains the same number of vectors).
   */
  std::vector<std::deque<BoltVector>> _buffers;
};

}  // namespace thirdai::dataset