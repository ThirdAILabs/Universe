#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace thirdai::dataset {

struct Coordinates {
  uint32_t column;
  uint32_t sample_id;
};

using ForEachVectorFunctor = std::function<void(Coordinates, BoltVector&&)>;

class BatchBuffer {
 public:
  BatchBuffer()
      : _size(0),
        // Starting ID of first batch is 0
        _start_ids({0}) {}

  BatchBuffer(BatchBuffer&& other) noexcept;

  BatchBuffer& operator=(BatchBuffer&& other) noexcept;

  void add(std::vector<BoltBatch>&& batch);

  uint32_t size() const { return _size; }

  bool empty() const { return _size == 0; }

  uint32_t numColumns() const { return _batches.size(); }

  void forEachVector(const ForEachVectorFunctor& functor);

 private:
  static void validateBatchSize(const std::vector<BoltBatch>& batch,
                                uint32_t batch_size);

  void validateBatchColumns(const std::vector<BoltBatch>& batch);

  uint32_t batchSize(uint32_t batch_id) const {
    return _batches.front().at(batch_id).getBatchSize();
  }

  uint32_t _size;
  std::vector<uint32_t> _start_ids;
  std::vector<std::vector<BoltBatch>> _batches;
};

class Shuffler {
 public:
  explicit Shuffler(bool shuffle, uint32_t seed)
      : _gen(seed), _shuffle(shuffle) {}

  void add(std::vector<BoltBatch>&& batch) { _buffer.add(std::move(batch)); }

  uint32_t size() const { return _buffer.size(); }

  std::optional<std::vector<BoltDatasetPtr>> datasets(uint32_t batch_size);

 private:
  static std::vector<std::vector<BoltBatch>> tidyBatches(BatchBuffer&& buffer,
                                                         uint32_t batch_size,
                                                         bool shuffle,
                                                         std::mt19937& gen);

  static std::vector<uint32_t> permute(uint32_t size, bool shuffle,
                                       std::mt19937& gen);

  static std::vector<std::vector<BoltBatch>> allocateTidyBatches(
      const BatchBuffer& buffer, uint32_t batch_size);

  std::mt19937 _gen;
  bool _shuffle;
  BatchBuffer _buffer;
};

}  // namespace thirdai::dataset