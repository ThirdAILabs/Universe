#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace thirdai::dataset {

// struct Coordinates {
//   Coordinates(uint32_t batch_id, uint32_t vec_id)
//       : batch_id(batch_id), vec_id(vec_id) {}

//   const uint32_t batch_id;
//   const uint32_t vec_id;
// };

class Shuffler {
 public:
  explicit Shuffler(bool shuffle, uint32_t seed)
      : _gen(seed), _shuffle(shuffle), _buffer_size(0), _offsets({0}) {}

  void add(std::vector<BoltBatch>&& batch) {
    _buffer_size += batch.front().getBatchSize();
    _offsets.push_back(_buffer_size);
    _buffer.push_back(std::move(batch));
  }

  uint32_t size() const { return _buffer_size; }

  std::vector<BoltDatasetPtr> datasets(uint32_t batch_size,
                                       uint32_t max_batches) {
    // Equivalent to vector of bolt datasets
    std::vector<std::vector<BoltBatch>> shuffled_batches =
        shuffle(std::move(_buffer), batch_size);

    uint32_t num_returned =
        std::min<uint32_t>(max_batches, shuffled_batches.front().size());

    std::vector<BoltDatasetPtr> output(shuffled_batches.size());

    for (uint32_t dataset_id = 0; dataset_id < output.size(); dataset_id++) {
      output[dataset_id] = std::make_shared<BoltDataset>(
          std::move(shuffled_batches[dataset_id]));
    }

    _buffer.clear();
    _buffer_size = 0;
    _offsets = {0};
    for (uint32_t remain_id = num_returned; remain_id < shuffled_batches.size();
         remain_id++) {
      std::vector<BoltBatch> batch(shuffled_batches[remain_id].size());
      for (uint32_t column_id = 0; column_id < batch.size(); column_id++) {
        batch[column_id] = std::move(shuffled_batches[column_id][remain_id]);
      }
      _buffer.push_back(std::move(batch));
      _buffer_size += _buffer.front().back().getBatchSize();
      _offsets.push_back(_buffer_size);
    }
    return output;
  }

  std::vector<std::vector<BoltBatch>> shuffle(
      std::vector<std::vector<BoltBatch>>&& buffer, uint32_t batch_size) {
    std::vector<uint32_t> permutation(_buffer_size);
    std::iota(permutation.begin(), permutation.end(), 0);
    if (_shuffle) {
      std::shuffle(permutation.begin(), permutation.end(), _gen);
    }

    uint32_t n_columns = buffer.front().size();
    uint32_t n_shuffled_batches = (_buffer_size + batch_size - 1) / batch_size;
    uint32_t last_batch_size = _buffer_size % batch_size;

    std::vector<std::vector<BoltBatch>> shuffled_batches(
        n_columns,
        std::vector<BoltBatch>(n_shuffled_batches, BoltBatch(batch_size)));

    for (auto& batch_list : shuffled_batches) {
      batch_list.back() = BoltBatch(last_batch_size);
    }

#pragma omp parallel for default(none) \
    shared(buffer, shuffled_batches, permutation, batch_size)
    for (uint32_t batch_id = 0; batch_id < buffer.size(); batch_id++) {
      for (uint32_t column_id = 0; column_id < buffer[batch_id].size();
           column_id++) {
        auto& unshuffled_batch = buffer[batch_id][column_id];
        for (uint32_t vec_id = 0; vec_id < unshuffled_batch.getBatchSize();
             vec_id++) {
          uint32_t sample_id = _offsets[batch_id] + vec_id;
          uint32_t shuffled_sample_id = permutation[sample_id];
          uint32_t shuffled_batch_id = shuffled_sample_id / batch_size;
          uint32_t shuffled_vec_id = shuffled_batch_id % batch_size;
          shuffled_batches[column_id][shuffled_batch_id][shuffled_vec_id] =
              std::move(buffer[batch_id][column_id][vec_id]);
        }
      }
    }

    return shuffled_batches;
  }

 private:
  std::mt19937 _gen;
  bool _shuffle;
  uint32_t _buffer_size;
  std::vector<uint32_t> _offsets;
  std::vector<std::vector<BoltBatch>> _buffer;
};

}  // namespace thirdai::dataset