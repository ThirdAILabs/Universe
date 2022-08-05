#pragma once

#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <atomic>
#include <cstddef>
#include <unordered_map>

namespace thirdai::dataset {

template <typename ELEMENT_T>
class AtomicCircularBuffers {
 public:
  explicit AtomicCircularBuffers(size_t n_buffers, size_t size)
      : _n_buffers(n_buffers),
        _size(size),
        _insert_idxs(n_buffers),
        _buffer(n_buffers * size) {}

  void refresh() {
    for (auto& idx : _insert_idxs) {
      size_t new_idx = idx.load() % _buffer.size();
      idx = new_idx;
    }
  }

  void insert(uint32_t buffer_idx, ELEMENT_T&& new_elem) {
    auto insert_idx = _insert_idxs[buffer_idx].fetch_add(1);
    _buffer[buffer_idx * _size + (insert_idx % _size)] =
        std::move(new_elem);
  }

  const std::vector<ELEMENT_T>& view() { return _buffer; }

  size_t startIdx(uint32_t buffer_idx) const { return _size * buffer_idx; }

  size_t endIdx(uint32_t buffer_idx) const { return _size * (buffer_idx + 1); }

  size_t n_buffers() const { return _n_buffers; }

 private:
  const size_t _n_buffers;
  const size_t _size;
  std::vector<std::atomic_size_t> _insert_idxs;
  std::vector<ELEMENT_T> _buffer;
};

struct CategoricalHistory {
  uint32_t timestamp = 0;
  // We don't store the string itself because we eventually need to
  // convert it to integers to vectorize anyway
  uint32_t uid = 0;
};

class CategoricalHistoryIndex {
 public:
  CategoricalHistoryIndex(uint32_t n_ids, uint32_t n_categories,
                          size_t buffer_size)
      : _categorical_map(n_categories), _index(n_ids, buffer_size) {}

  void index(uint32_t id, uint32_t timestamp,
             std::string_view categorical_attr) {
    if (id >= _index.n_buffers()) {
      std::cout << "WARNING: Tried to index with id " << id
                << " into a CategoricalHistoryIndex for " << _index.n_buffers()
                << " classes. Treated as a no-op." << std::endl;
      return;
    }
    auto cat_id = _categorical_map.classToUid(categorical_attr);
    _index.insert(id, {timestamp, cat_id});
  }

  void refresh() { _index.refresh(); }

  const std::vector<CategoricalHistory>& view() { return _index.view(); }

  size_t startIdx(uint32_t id) const {
    if (id >= _index.n_buffers()) {
      std::cout << "WARNING: Tried to get buffer for id " << id
                << " from a CategoricalHistoryIndex for " << _index.n_buffers()
                << " classes. Treated as a no-op." << std::endl;
      return 0;
    }
    return _index.startIdx(id);
  }

  size_t endIdx(uint32_t id) const {
    if (id >= _index.n_buffers()) {
      std::cout << "WARNING: Tried to get buffer for id " << id
                << " from a CategoricalHistoryIndex for " << _index.n_buffers()
                << " classes. Treated as a no-op." << std::endl;
      return 0;
    }
    return _index.endIdx(id);
  }

  size_t featureDim() const { return _categorical_map.featureDim(); }

 private:
  StringToUidMap _categorical_map;
  AtomicCircularBuffers<CategoricalHistory> _index;
};

}  // namespace thirdai::dataset