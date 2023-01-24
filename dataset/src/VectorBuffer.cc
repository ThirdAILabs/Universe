
#include "VectorBuffer.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <ctime>
#include <deque>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>

namespace thirdai::dataset {

void VectorBuffer::insertBatch(std::vector<BoltBatch>&& batches, bool shuffle) {
  checkConsistentBatchSize(batches);

  size_t batch_size = batches.at(0).getBatchSize();

  initializeBuffersIfNeeded(batches);

  for (uint32_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
    for (auto& vector : batches.at(buffer_id)) {
      _buffers.at(buffer_id).push_back(std::move(vector));
    }
  }

  if (shuffle) {
    swapShuffle(_buffers, /* batch_size_added = */ batch_size, _gen);
  }
}

std::optional<std::vector<BoltBatch>> VectorBuffer::popBatch(
    size_t target_batch_size) {
  if (empty()) {
    return std::nullopt;
  }

  std::vector<std::vector<BoltVector>> vecs_to_return(_buffers.size());
  for (size_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
    auto& buffer = _buffers.at(buffer_id);
    for (size_t vec_id = 0; vec_id < target_batch_size; vec_id++) {
      if (buffer.empty()) {
        break;
      }
      vecs_to_return.at(buffer_id).push_back(std::move(buffer.front()));
      buffer.pop_front();
    }
  }

  std::vector<BoltBatch> batches_to_return(_buffers.size());
  for (size_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
    batches_to_return.at(buffer_id) =
        BoltBatch(std::move(vecs_to_return.at(buffer_id)));
  }

  return batches_to_return;
}

/**
 * Pops min(num_batches, size()) batches into a vector of vector of
 * BoltBatch (with batch size target_batch_size, except possible the last
 * entry in each vector which may be smaller)
 */
std::vector<std::vector<BoltBatch>> VectorBuffer::popBatches(
    size_t num_batches, size_t target_batch_size) {
  std::vector<std::vector<BoltBatch>> exported_batch_lists(_buffers.size());
  while (!empty() && exported_batch_lists.at(0).size() < num_batches) {
    auto next_batches = *popBatch(target_batch_size);
    assert(next_batches.size() == _buffers.size());

    for (size_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
      exported_batch_lists.at(buffer_id).push_back(
          std::move(next_batches.at(buffer_id)));
    }
  }
  return exported_batch_lists;
}

void VectorBuffer::initializeBuffersIfNeeded(
    const std::vector<BoltBatch>& batches) {
  if (_buffers.empty()) {
    _buffers = std::vector<std::deque<BoltVector>>(batches.size());
  }

  if (_buffers.size() != batches.size()) {
    std::stringstream error_ss;
    error_ss << "[VectorBuffer::insertBatch] Attempted to insert "
                "a different number of corresponding batches than originally "
                "inserted into the buffer (originally inserted "
             << _buffers.size() << ", trying to insert " << batches.size()
             << ").";
    throw std::runtime_error(error_ss.str());
  }
}

inline void VectorBuffer::checkConsistentBatchSize(
    const std::vector<BoltBatch>& batches) {
  if (batches.empty()) {
    throw std::runtime_error(
        "[VectorBuffer::insertBatch] Expected at least one "
        "batch to be inserted for shuffling but found 0.");
  }
  uint32_t first_data_batch_size = batches.at(0).getBatchSize();
  for (uint32_t i = 1; i < batches.size(); i++) {
    if (batches.at(i).getBatchSize() != first_data_batch_size) {
      std::stringstream error_ss;
      error_ss << "[VectorBuffer::insertBatch] Attempted to insert "
                  "corresponding batches with different sizes (one size = "
               << first_data_batch_size
               << ", the other size = " << batches.at(i).getBatchSize() << ").";
      throw std::runtime_error(error_ss.str());
    }
  }
}

inline void VectorBuffer::swapShuffle(
    std::vector<std::deque<BoltVector>>& buffers, size_t batch_size_added,
    std::mt19937& gen) {
  assert(buffers.at(0).size() > 0);

  size_t n_vecs = buffers.at(0).size();
  size_t n_old_vecs = n_vecs - batch_size_added;
  std::uniform_int_distribution<> dist(
      0, n_vecs - 1);  // Accepts a closed interval

  for (size_t new_vec_id = n_old_vecs; new_vec_id < n_vecs; new_vec_id++) {
    size_t swap_with = dist(gen);
    for (auto& buffer : buffers) {
      std::swap(buffer.at(new_vec_id), buffer.at(swap_with));
    }
  }
}

}  // namespace thirdai::dataset