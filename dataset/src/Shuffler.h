#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace thirdai::dataset {

class Shuffler {
 public:
  explicit Shuffler(bool shuffle, uint32_t seed)
      : _gen(seed), _shuffle(shuffle), _buffer_size(0), _offsets({0}) {}

  void add(std::vector<BoltBatch>&& batch);

  uint32_t size() const { return _buffer_size; }

  std::vector<BoltDatasetPtr> datasets(uint32_t batch_size,
                                       uint32_t max_batches);

  std::vector<std::vector<BoltBatch>> shuffle(
      std::vector<std::vector<BoltBatch>>&& buffer, uint32_t batch_size);

 private:
  std::mt19937 _gen;
  bool _shuffle;
  uint32_t _buffer_size;
  std::vector<uint32_t> _offsets;
  std::vector<std::vector<BoltBatch>> _buffer;
};

}  // namespace thirdai::dataset