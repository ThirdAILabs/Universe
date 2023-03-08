#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <cstddef>
#include <deque>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

/**
 * Hold me closer tidy baaaatcheeeerr
 */
class TidyBatcher {
 public:
  explicit TidyBatcher(std::mt19937& gen)
      : _used(false), _size(0), _start_ids({0}), _gen(gen) {}

  void add(std::vector<BoltBatch>&& batch);

  std::optional<std::vector<std::vector<BoltBatch>>> batches(size_t batch_size,
                                                             bool shuffle);

  size_t size() const { return _size; }

 private:
  size_t numColumns() const { return _batches.front().size(); }

  std::vector<std::vector<BoltBatch>> allocatePoppedBatches(size_t batch_size);

  std::vector<uint32_t> ordering(bool shuffle);

  bool _used;
  size_t _size;
  std::vector<size_t> _start_ids;
  std::vector<std::vector<BoltBatch>> _batches;
  std::mt19937& _gen;
};

}  // namespace thirdai::dataset