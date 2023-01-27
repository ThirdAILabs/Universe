
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

void VectorBuffer::insert(std::vector<BoltVector>&& vectors) {
  verifyCorrectNumberOfVectors(vectors);

  for (uint32_t buffer_id = 0; buffer_id < _buffers.size(); buffer_id++) {
    _buffers.at(buffer_id).push_back(std::move(vectors.at(buffer_id)));
  }

  if (_shuffle) {
    shuffleNewVectors();
  }
}

std::optional<std::vector<BoltVector>> VectorBuffer::pop() {
  if (empty()) {
    return std::nullopt;
  }

  std::vector<BoltVector> vecs_to_return;
  for (auto& buffer : _buffers) {
    vecs_to_return.push_back(std::move(buffer.front()));
    buffer.pop_front();
  }

  return vecs_to_return;
}

void VectorBuffer::verifyCorrectNumberOfVectors(
    const std::vector<BoltVector>& vectors) {
  assert(!_buffers.empty());

  if (_buffers.size() != vectors.size()) {
    std::stringstream error_ss;
    error_ss
        << "[VectorBuffer::insert] Attempted to insert "
           "a different number of corresponding vectors than the "
           "buffer was configured with (the buffer was configured to track "
        << _buffers.size() << "datasets, trying to insert " << vectors.size()
        << ").";
    throw std::runtime_error(error_ss.str());
  }
}

void VectorBuffer::shuffleNewVectors() {
  assert(_buffers.at(0).size() > 0);

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