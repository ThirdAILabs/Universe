#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace thirdai::dataset {

template <typename ELEMENT_T>
class CircularBuffer {
 public:
  CircularBuffer() : _insert_idx(0), _pop_idx(0), _size(0), _n_popped(0) {}

  void insert(ELEMENT_T&& new_elem) {
    if (_size == _buffer.size() && _n_popped > 0) {
      throw std::runtime_error(
          "[CircularBuffer::insert] Attempted to resize "
          "non-contiguous buffer. CircularBuffer cannot be resized after "
          "popping an element.");
    }

    if (_size == _buffer.size()) {
      _buffer.push_back(std::move(new_elem));
      _insert_idx = 0;
    } else {
      _buffer[_insert_idx] = std::move(new_elem);
      _insert_idx = (_insert_idx + 1) % _buffer.size();
    }
    _size++;
  }

  std::optional<ELEMENT_T> pop() {
    if (_size == 0) {
      return {};
    }

    ELEMENT_T elem = std::move(_buffer[_pop_idx]);
    _pop_idx = (_pop_idx + 1) % _buffer.size();
    _size--;
    _n_popped++;

    return elem;
  }

  ELEMENT_T& at(size_t i) {
    if (i >= _size) {
      throw std::invalid_argument("[CircularBuffer::at] Index out of range.");
    }
    size_t index = (_pop_idx + i) % _buffer.size();
    return _buffer[index];
  }

  size_t size() const { return _size; }

  bool empty() const { return _size == 0; }

  std::vector<ELEMENT_T> exportContiguousBuffer() {
    if (_n_popped > 0) {
      throw std::runtime_error(
          "[CircularBuffer::exportContiguousBuffer] Attempted to export "
          "non-contiguous buffer. Buffer cannot be exported after popping an "
          "element.");
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
  std::vector<ELEMENT_T> _buffer;
  size_t _insert_idx;
  size_t _pop_idx;
  size_t _size;
  uint64_t _n_popped;
};

}  // namespace thirdai::dataset