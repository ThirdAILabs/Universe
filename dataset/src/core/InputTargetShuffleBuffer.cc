#include "InputTargetShuffleBuffer.h"
#include <cstdlib>
#include <stdexcept>

namespace thirdai::dataset {

InputTargetShuffleBuffer::InputTargetShuffleBuffer(size_t batch_size, size_t num_buffer_batches, bool has_target)
: _first_elem_idx(0),
  _new_batch_start_idx(0),
  _size(0),
  _batch_size(batch_size), 
  _inputs(num_buffer_batches * batch_size),
  _targets() {

  if (has_target) {
    _targets = std::vector<bolt::BoltVector>(num_buffer_batches * batch_size);
  }
}

OptionalInputTargetBatch InputTargetShuffleBuffer::nextBatch() {
  if (_size == 0) {
    return {};
  }

  bolt::BoltBatch input_batch = makeBatchFrom(_inputs);
  std::optional<bolt::BoltBatch> target_batch;
  if (_targets.has_value()) {
    target_batch = makeBatchFrom(_targets.value());
  }

  // Update state
  _first_elem_idx = wrap(_first_elem_idx + input_batch.getBatchSize());
  _size -= input_batch.getBatchSize();

  // Build batches
  return {{std::move(input_batch), std::move(target_batch)}};
}

void InputTargetShuffleBuffer::addBatch(ProcessedBatch &&batch, bool shuffle) {
  auto cur_batch_size = batch.first.size();
  if (_inputs.size() - _size < cur_batch_size) {
    throw std::invalid_argument("Not enough space in InputTargetBuffer for a new batch.");
  }

  for (uint32_t i = 0; i < batch.first.size(); i++) {
    _inputs.at(_new_batch_start_idx + i) = std::move(batch.first.at(i));
  }
  if (batch.second.has_value()) {
    for (uint32_t i = 0; i < batch.second->size(); i++) {
      _targets.value().at(_new_batch_start_idx + i) = std::move(batch.second->at(i));
    } 
  }

  if (shuffle) {
    this->shuffle(cur_batch_size);
  }

  // Update state
  _new_batch_start_idx = wrap(_new_batch_start_idx + cur_batch_size);
  _size += cur_batch_size;
}

// Helper functions

void InputTargetShuffleBuffer::shuffle(size_t latest_batch_size) {
  auto new_size = _size + latest_batch_size;

  for (uint32_t i = 0; i < latest_batch_size; i++) {
    uint32_t cur_idx = wrap(_new_batch_start_idx + i);
    auto swap_vec = std::rand() % new_size;
    // We don't swap if swap_vec is in the current batch
    // because position within a batch doesn't matter.
    // Additionally, if we swap with another element in 
    // the same batch, that second element will not have
    // a chance to be in an earlier batch.
    if (swap_vec < _size) {
      auto swap_idx = wrap(_first_elem_idx + swap_vec);
      swapElemsAtIndices(_inputs, cur_idx, swap_idx);
      if (_targets.has_value()) {
        swapElemsAtIndices(_targets.value(), cur_idx, swap_idx);
      }
    }
  }
}

bolt::BoltBatch InputTargetShuffleBuffer::makeBatchFrom(std::vector<bolt::BoltVector>& vector_buf) {
  auto cur_batch_size = std::min(_batch_size, _size);
  std::vector<bolt::BoltVector> batch_vecs(cur_batch_size);
  auto buf_idx = _first_elem_idx;
  for (uint32_t i = 0; i < cur_batch_size; i++) {
    batch_vecs[i] = std::move(vector_buf.at(buf_idx));
    buf_idx = wrap(buf_idx + 1);
  }
  return bolt::BoltBatch(std::move(batch_vecs));
}

inline uint32_t InputTargetShuffleBuffer::wrap(uint32_t idx) const {
  if (idx >= _inputs.size()) {
    idx -= _inputs.size();
  }
  return idx;
}

} // namespace thirdai::dataset