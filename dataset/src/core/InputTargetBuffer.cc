#include "InputTargetBuffer.h"
#include <cstdlib>
#include <stdexcept>

namespace thirdai::dataset {

InputTargetBuffer::InputTargetBuffer(size_t batch_size, size_t num_buffer_batches, bool has_target)
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

OptionalInputTargetBatch InputTargetBuffer::nextBatch() {
  if (_size == 0) {
    return {};
  }

  // Inputs
  std::vector<bolt::BoltVector> input_batch_vecs(_batch_size);
  auto input_buf_idx = _first_elem_idx;
  for (uint32_t i = 0; i < _batch_size; i++) {
    input_batch_vecs[i] = std::move(_inputs.at(input_buf_idx));
    input_buf_idx = nextElemIdx(input_buf_idx);
  }
  bolt::BoltBatch input_batch(std::move(input_batch_vecs));

  // Targets
  std::optional<bolt::BoltBatch> target_batch;
  if (_targets.has_value()) {
    std::vector<bolt::BoltVector> target_batch_vecs;
    target_batch_vecs = std::vector<bolt::BoltVector>(_batch_size);
    auto target_buf_idx = _first_elem_idx;
    for (uint32_t i = 0; i < _batch_size; i++) {
      target_batch_vecs[i] = std::move(_targets->at(target_buf_idx));
      target_buf_idx = nextElemIdx(target_buf_idx);
    }
    target_batch = bolt::BoltBatch(std::move(target_batch_vecs));
  }

  // Update member variables
  _first_elem_idx = input_buf_idx;
  _size -= _batch_size;

  // Build batches
  return {{std::move(input_batch), std::move(target_batch)}};
}

void InputTargetBuffer::initiateNewBatch() {
  if (_inputs.size() - _size < _batch_size) {
    throw std::invalid_argument("Not enough space in InputTargetBuffer for a new batch.");
  }
}

void InputTargetBuffer::addNewBatchInputVec(uint32_t idx, bolt::BoltVector&& input_vec) {
  _inputs[_new_batch_start_idx + idx] = input_vec;
}

void InputTargetBuffer::addNewBatchTargetVec(uint32_t idx, bolt::BoltVector&& target_vec) {
  _targets.value().at(_new_batch_start_idx + idx) = target_vec;
}

void InputTargetBuffer::finalizeNewBatch(bool shuffle) {
  auto new_size = _size + _batch_size;

  // shuffle if needed.
  if (shuffle) {
    // We don't need to do cur_idx % _inputs.size() (buffer size) 
    // because size of _inputs is always a multiple of batch_size.
    for (uint32_t cur_idx = _new_batch_start_idx; cur_idx < _new_batch_start_idx + _batch_size; cur_idx++) {
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

  // update member variables
  _new_batch_start_idx = wrap(_new_batch_start_idx + _batch_size);
  _size += _batch_size;
}

// Helper functions

inline uint32_t InputTargetBuffer::nextElemIdx(uint32_t prev_elem_idx) const {
  auto idx = prev_elem_idx + 1;
  return wrap(idx);
}

inline uint32_t InputTargetBuffer::wrap(uint32_t idx) const {
  if (idx >= _inputs.size()) {
    idx -= _inputs.size();
  }
  return idx;
}

} // namespace thirdai::dataset