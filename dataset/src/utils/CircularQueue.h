#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::dataset {

/**
 * A circular queue that dynamically grows until the first element is popped.
 * At that point, it becomes a bounded circular queue with a size limit
 * equal to the size at the time when the first element was popped.
 */
template <typename ELEMENT_T>
class CircularQueue {
 public:
  CircularQueue() : _insert_idx(0), _pop_idx(0), _size(0), _n_popped(0) {}

  /**
   * Inserts an element into the circular queue. If the queue is full and
   * no element had been popped, the circular buffer gets resized.
   * If the first element had been popped, the queue's size is bounded,
   * so this method throws an error if the queue is full.
   */
  void insert(ELEMENT_T&& new_elem) {
    if (bufferIsFull() && _n_popped > 0) {
      throw std::runtime_error(
          "[CircularQueue::insert] Attempted to insert into a full bounded "
          "queue. A CircularQueue's size is bounded once an element has been "
          "popped.");
    }

    if (bufferIsFull()) {
      _buffer.push_back(std::move(new_elem));
      _insert_idx = 0;
    } else {
      _buffer[_insert_idx] = std::move(new_elem);
      _insert_idx = rotateRight(_insert_idx, /* by = */ 1);
    }
    _size++;
  }

  /**
   * Returns the first element in the queue. Returns nullopt if queue is empty.
   */
  std::optional<ELEMENT_T> pop() {
    if (_size == 0) {
      return {};
    }

    ELEMENT_T elem = std::move(_buffer[_pop_idx]);
    _pop_idx = rotateRight(_pop_idx, /* by = */ 1);
    _size--;
    _n_popped++;

    return elem;
  }

  /**
   * Returns the i-th element in the queue.
   */
  ELEMENT_T& at(size_t i) {
    if (i >= _size) {
      throw std::invalid_argument("[CircularQueue::at] Index out of range.");
    }
    size_t index = rotateRight(_pop_idx, /* by = */ i);
    return _buffer[index];
  }

  /**
   * Returns the last element in the queue.
   */
  ELEMENT_T& last() {
    if (_size == 0) {
      throw std::runtime_error("[CircularQueue::last] Queue is emtpy.");
    }
    size_t index = rotateRight(_pop_idx, /* by = */ _size - 1);
    return _buffer[index];
  }

  /**
   * Returns the current number of elements in the queue.
   */
  size_t size() const { return _size; }

  /**
   * Returns whether the queue is currently empty.
   */
  bool empty() const { return _size == 0; }

  /**
   * Exports the circular queue's buffer as an std::vector<ELEMENT_T>.
   * Throws an error if the buffer is not contiguous.
   */
  std::vector<ELEMENT_T> exportContiguousBuffer() {
    // A buffer is contiguous if an only if it is full.
    if (!bufferIsFull()) {
      throw std::runtime_error(
          "[CircularQueue::exportContiguousBuffer] Attempted to export "
          "non-contiguous buffer. This is not allowed because there may "
          "be invalid elements in the buffer. A buffer may cease to be "
          "contiguous if elements are popped and not replenished.");
    }

    auto buffer = std::move(_buffer);

    _buffer = std::vector<ELEMENT_T>();
    _insert_idx = 0;
    _pop_idx = 0;
    _size = 0;
    _n_popped = 0;

    return buffer;
  }

 private:
  bool bufferIsFull() { return _size == _buffer.size(); }

  size_t rotateRight(size_t idx, size_t by) {
    return (idx + by) % _buffer.size();
  }

  std::vector<ELEMENT_T> _buffer;
  size_t _insert_idx;
  size_t _pop_idx;
  size_t _size;
  uint64_t _n_popped;
};

}  // namespace thirdai::dataset