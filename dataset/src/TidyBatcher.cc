#include "TidyBatcher.h"

namespace thirdai::dataset {

void TidyBatcher::add(std::vector<BoltBatch>&& batch) {
  if (_used) {
    throw std::invalid_argument(
        "BatchBuffer cannot be reused after calling the batches() method.");
  }
  _size += batch.front().getBatchSize();
  _start_ids.push_back(_size);
  _batches.push_back(std::move(batch));
}

std::optional<std::vector<std::vector<BoltBatch>>> TidyBatcher::batches(
    size_t batch_size, bool shuffle) {
  if (_used) {
    throw std::invalid_argument(
        "BatchBuffer cannot be reused after calling the batches() method.");
  }
  _used = true;
  if (_batches.empty()) {
    return std::nullopt;
  }

  auto tidy = allocateTidyBatches(batch_size);
  auto permutation = ordering(shuffle);

#pragma omp parallel for default(none) \
    shared(_batches, _start_ids, batch_size, tidy, permutation)
  for (size_t batch_id = 0; batch_id < _batches.size(); batch_id++) {
    for (size_t vec_id = 0; vec_id < _batches[batch_id].front().getBatchSize();
         vec_id++) {
      size_t id = _start_ids[batch_id] + vec_id;
      size_t tidy_id = permutation[id];

      size_t tidy_batch_id = tidy_id / batch_size;
      size_t tidy_vec_id = tidy_id % batch_size;

      for (size_t column_id = 0; column_id < _batches.front().size();
           column_id++) {
        tidy[column_id][tidy_batch_id][tidy_vec_id] =
            std::move(_batches[batch_id][column_id][vec_id]);
      }
    }
  }

  return tidy;
}

std::vector<std::vector<BoltBatch>> TidyBatcher::allocateTidyBatches(
    size_t batch_size) {
  size_t num_batches = (_size + batch_size - 1) / batch_size;
  size_t last_batch_size = _size % batch_size;
  if (last_batch_size == 0) {
    last_batch_size = batch_size;
  }

  std::vector<std::vector<BoltBatch>> batches(
      numColumns(), std::vector<BoltBatch>(num_batches));

#pragma omp parallel for default(none) \
    shared(num_batches, batch_size, last_batch_size, batches)
  for (uint32_t batch_id = 0; batch_id < num_batches; batch_id++) {
    uint32_t this_batch_size =
        batch_id == num_batches - 1 ? last_batch_size : batch_size;
    for (auto& batch_list : batches) {
      batch_list[batch_id] = BoltBatch(this_batch_size);
    }
  }

  return batches;
}

std::vector<uint32_t> TidyBatcher::ordering(bool shuffle) {
  std::vector<uint32_t> permutation(_size);
  std::iota(permutation.begin(), permutation.end(), 0);
  if (shuffle) {
    std::shuffle(permutation.begin(), permutation.end(), _gen);
  }
  return permutation;
}

}  // namespace thirdai::dataset