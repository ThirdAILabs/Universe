#include "Shuffler.h"
#include <bolt_vector/src/BoltVector.h>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

BatchBuffer::BatchBuffer(BatchBuffer&& other) noexcept
    : _size(other._size),
      _start_ids(std::move(other._start_ids)),
      _batches(std::move(other._batches)) {
  other._size = 0;
}

BatchBuffer& BatchBuffer::operator=(BatchBuffer&& other) noexcept {
  _size = other._size;
  _start_ids = std::move(other._start_ids);
  _batches = std::move(other._batches);

  other._size = 0;

  return *this;
}

void BatchBuffer::add(std::vector<BoltBatch>&& batch) {
  uint32_t batch_size = batch.front().getBatchSize();
  validateBatchSize(batch, batch_size);
  validateBatchColumns(batch);

  _size += batch.front().getBatchSize();
  _start_ids.push_back(_size);

  if (_batches.empty()) {
    _batches.resize(batch.size());
  }

  for (uint32_t column = 0; column < _batches.size(); column++) {
    _batches[column].push_back(std::move(batch[column]));
  }
}

void BatchBuffer::forEachVector(const ForEachVectorFunctor& functor) {
#pragma omp parallel for default(none) \
    shared(size, numColumns, batchSize, _start_ids, functor, _batches)
  if (_batches.empty()) {
    return;
  }
  for (uint32_t batch_id = 0; batch_id < _batches.front().size(); batch_id++) {
    for (uint32_t column = 0; column < numColumns(); column++) {
      for (uint32_t vec_id = 0; vec_id < batchSize(batch_id); vec_id++) {
        uint32_t sample_id = _start_ids[batch_id] + vec_id;

        Coordinates coords;
        coords.column = column;
        coords.sample_id = sample_id;
        functor(coords, std::move(_batches[column][batch_id][vec_id]));
      }
    }
  }
}

void BatchBuffer::validateBatchSize(const std::vector<BoltBatch>& batch,
                                    uint32_t batch_size) {
  for (const auto& column : batch) {
    if (column.getBatchSize() != batch_size) {
      throw std::invalid_argument(
          "Saw a batch with inconsistent sizes across columns.");
    }
  }
}

void BatchBuffer::validateBatchColumns(const std::vector<BoltBatch>& batch) {
  if (!_batches.empty() && batch.size() != _batches.size()) {
    throw std::invalid_argument(
        "Saw a batch with inconsistent number of columns.");
  }
}

std::optional<std::vector<BoltDatasetPtr>> Shuffler::datasets(
    uint32_t batch_size) {
  if (_buffer.empty()) {
    return std::nullopt;
  }
  auto tidy_batches =
      tidyBatches(std::move(_buffer), batch_size, _shuffle, _gen);

  uint32_t num_batches = tidy_batches.front().size();

  std::vector<BoltDatasetPtr> output(tidy_batches.size());
  for (uint32_t dataset_id = 0; dataset_id < output.size(); dataset_id++) {
    tidy_batches[dataset_id].resize(num_batches);
    output[dataset_id] =
        std::make_shared<BoltDataset>(std::move(tidy_batches[dataset_id]));
  }

  return output;
}

std::vector<std::vector<BoltBatch>> Shuffler::tidyBatches(BatchBuffer&& buffer,
                                                          uint32_t batch_size,
                                                          bool shuffle,
                                                          std::mt19937& gen) {
  auto permutation = permute(buffer.size(), shuffle, gen);
  auto tidy_batches = allocateTidyBatches(buffer, batch_size);

  buffer.forEachVector([&permutation, &tidy_batches, &batch_size](
                           Coordinates coords, BoltVector&& vector) {
    uint32_t sample_id = permutation[coords.sample_id];
    uint32_t batch_id = sample_id / batch_size;
    uint32_t vec_id = sample_id % batch_size;

    tidy_batches[coords.column][batch_id][vec_id] = std::move(vector);
  });

  return tidy_batches;
}

std::vector<uint32_t> Shuffler::permute(uint32_t size, bool shuffle,
                                        std::mt19937& gen) {
  std::vector<uint32_t> permutation(size);
  std::iota(permutation.begin(), permutation.end(), 0);
  if (shuffle) {
    std::shuffle(permutation.begin(), permutation.end(), gen);
  }
  return permutation;
}

std::vector<std::vector<BoltBatch>> Shuffler::allocateTidyBatches(
    const BatchBuffer& buffer, uint32_t batch_size) {
  uint32_t n_shuffled_batches = (buffer.size() + batch_size - 1) / batch_size;
  uint32_t last_batch_size = buffer.size() % batch_size;

  std::vector<std::vector<BoltBatch>> tidy_batches(
      buffer.numColumns(), std::vector<BoltBatch>(n_shuffled_batches));

#pragma omp parallel for default(none) \
    shared(n_shuffled_batches, tidy_batches, batch_size)
  for (uint32_t shuffled_batch_id = 0;
       shuffled_batch_id < n_shuffled_batches - 1; shuffled_batch_id++) {
    for (auto& batch_list : tidy_batches) {
      batch_list[shuffled_batch_id] = BoltBatch(batch_size);
    }
  }
  for (auto& batch_list : tidy_batches) {
    batch_list.back() = BoltBatch(last_batch_size);
  }

  return tidy_batches;
}

}  // namespace thirdai::dataset