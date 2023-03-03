#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <vector>

namespace thirdai::dataset {

class Shuffler {
 public:
  explicit Shuffler(uint32_t batch_size) : _batch_size(batch_size) {}

  void add(std::vector<std::vector<BoltVector>>&& vectors) {
    _buffer.push_back(std::move(vectors));
  }

  std::vector<BoltDatasetPtr> datasets(uint32_t max_batches) {
    // Equivalent to vector of bolt datasets
    std::vector<std::vector<BoltBatch>> shuffled_batches =
        shuffle(std::move(_buffer));

    uint32_t num_returned =
        std::min<uint32_t>(max_batches, shuffled_batches.front().size());

    std::vector<BoltDatasetPtr> output(shuffled_batches.size());

    for (uint32_t dataset_id = 0; dataset_id < output.size(); dataset_id++) {
      std::move(shuffled_batches[dataset_id].begin(),
                shuffled_batches[dataset_id].begin() + num_returned,
                output[dataset_id]->begin());
    }

    _buffer.clear();
    for (uint32_t remain_id = num_returned) }

  std::vector<std::vector<BoltBatch>> shuffle(
      std::vector<std::vector<std::vector<BoltVector>>>&& buffer) {}

 private:
  uint32_t _batch_size;
  std::vector<std::vector<std::vector<BoltVector>>> _buffer;
};

}  // namespace thirdai::dataset