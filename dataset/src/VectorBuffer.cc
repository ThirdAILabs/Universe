
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

void VectorBuffer::insert(std::vector<BoltVector>&& vectors, bool shuffle) {
  initializeBuffersIfNeeded(vectors);

  for (uint32_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
    _buffers.at(buffer_id).push_back(std::move(vectors.at(buffer_id)));
  }

  if (shuffle) {
    shuffleNewVectors();
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
    const std::vector<BoltVector>& vectors) {
  if (_buffers.empty()) {
    _buffers = std::vector<std::deque<BoltVector>>(vectors.size());
  }

  if (_buffers.size() != vectors.size()) {
    std::stringstream error_ss;
    error_ss << "[VectorBuffer::insertBatch] Attempted to insert "
                "a different number of corresponding vectors than originally "
                "inserted into the buffer (originally inserted "
             << _buffers.size() << ", trying to insert " << vectors.size()
             << ").";
    throw std::runtime_error(error_ss.str());
  }
}

void VectorBuffer::shuffleNewVectors() {
  assert(buffers.at(0).size() > 0);

  size_t buffer_size = _buffers.at(0).size();
  std::uniform_int_distribution<> dist(
      0, buffer_size - 1);  // Accepts a closed interval
  size_t swap_with = dist(_gen);

  for (auto& buffer : _buffers) {
    auto& new_vector = buffer.at(buffer_size - 1);
    auto& swap_vector = buffer.at(swap_with);
    std::swap(new_vector, swap_vector);
  }
}

}  // namespace thirdai::dataset