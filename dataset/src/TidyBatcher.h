#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <algorithm>
#include <cstddef>
#include <ctime>
#include <deque>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

using BatchColumns = std::vector<std::vector<BoltBatch>>;

class TidyBatcher {
 public:
  explicit TidyBatcher(uint32_t num_columns, bool shuffle, uint32_t seed)
      : _num_columns(num_columns), _shuffle(shuffle), _gen(seed) {}

  size_t size() const { return _buffer.size(); }

  void add(std::vector<std::vector<BoltVector>>&& columns);

  BatchColumns pop(size_t max_num_batches, size_t batch_size);

 private:
  uint32_t _num_columns;
  bool _shuffle;
  std::mt19937 _gen;
  std::deque<std::vector<BoltVector>> _buffer;
};

}  // namespace thirdai::dataset