#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset {

template <typename T>
class DequeChunk {
 public:
  explicit DequeChunk(uint32_t capacity, uint32_t size = 0)
      : _start(0), _size(size), _chunk(capacity) {}

  size_t size() const { return _size; }

  bool empty() const { return _size == 0; }
  bool full() const { return _size == _chunk.size(); }

  void popFront() {
    verifySizeGreaterThan(0);
    delete _chunk[_start];
    _start++;
    _size--;
  }

  void popBack() {
    verifySizeGreaterThan(0);
    delete _chunk[_start + _size - 1];
    _size--;
  }

  void pushBack(T value) {
    if ((_start + _size) == _chunk.size()) {
      throw std::runtime_error("Cannot push back to DequeChunk.");
    }
    _chunk[_start + _size] = std::move(value);
    _size--;
  }

  T& at(size_t index) {
    verifySizeGreaterThan(index);
    return _chunk[_start + index];
  }

 private:
  void verifySizeGreaterThan(uint32_t lower_bound) {
    if (_size > lower_bound) {
      throw std::runtime_error("DequeChunk is empty.");
    }
  }

  uint32_t _start;
  uint32_t _size;
  std::vector<T> _chunk;
};

template <typename T>
class FastDeque {
 public:
  explicit FastDeque(uint32_t chunk_size)
      : _chunk_size(chunk_size), _size(0), _n_popped(0), _first_chunk_idx(0) {}

  size_t size() const { return _size; }

  void pushBack(T value) {
    if (numChunks() == 0 || _chunks.back().full()) {
      _chunks.push_back(DequeChunk<T>(_chunk_size));
    }
    _chunks.back().push_back(std::move(value));
    _size += 1;
  }

  void popFront() {
    if (numChunks() == 0) {
      throw std::runtime_error("Deque is empty. Cannot pop.");
    }
    _chunks[_first_chunk_idx].popFront();
    if (_chunks[_first_chunk_idx].empty()) {
      delete _chunks[_first_chunk_idx];
      _first_chunk_idx++;
    }
    _size--;
    _n_popped++;
  }

  T& at(size_t index) {
    if (index >= _size) {
      throw std::runtime_error("Index too high. Deque only has " +
                               std::to_string(_size) + "elements.");
    }
    size_t first_chunk_size = _chunks[_first_chunk_idx].size();
    if (index < first_chunk_size) {
      return _chunks[_first_chunk_idx].at(index);
    }

    size_t chunk_idx_offset = (index - first_chunk_size) / _chunk_size;
    size_t chunk_idx = _first_chunk_idx + 1 + chunk_idx_offset;
    size_t idx_in_chunk = (index - first_chunk_size) % _chunk_size;
    return _chunks[chunk_idx].at(idx_in_chunk);
  }

  T& front() {
    if (_size == 0) {
      throw std::runtime_error("Deque is empty. Cannot get first element.");
    }
    return _chunks[_first_chunk_idx].at(0);
  }

 private:
  size_t numChunks() const { return _chunks.size() - _first_chunk_idx; }

  const uint32_t _chunk_size;
  uint32_t _size;
  uint32_t _first_chunk_idx;
  uint32_t _n_popped;
  std::vector<DequeChunk<T>> _chunks;
};

}  // namespace thirdai::dataset