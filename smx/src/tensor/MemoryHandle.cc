#include "MemoryHandle.h"
#include <optional>
#include <unordered_map>
#include <vector>

namespace thirdai::smx {

/**
 * While the allocations take <1ms even for large tensors, in some cases
 * deallocating large tensors was expensive. This solves that problem by instead
 * caching the allocated blocks and reusing them for future tensors with the
 * same size. This works because the sizes of tensors a model will allocate will
 * be repeated.
 *
 * Future improvements:
 *  1. Should the max size be based on some number of total bytes, rather than
 *     cache entries.
 *  2. Either add options for adjusting max entries or use std::array<...> to so
 *     nothing is on the heap.
 *  3. Cache eviction: remove smallest/oldest/largest or random when cache is
 *     full.
 */
class AllocationCache {
 public:
  using CacheEntry = std::pair<std::unique_ptr<uint8_t[]>, size_t>;

  void insert(std::unique_ptr<uint8_t[]> data, size_t size) {
    if (full() || size < _min_size_bytes) {
      return;
    }
    _cache[size].emplace_back(std::move(data), size);
    _curr_cache_entries++;
  }

  std::optional<CacheEntry> find(size_t size) {
    if (_cache.count(size) && !_cache.at(size).empty()) {
      CacheEntry out = std::move(_cache.at(size).back());
      _cache.at(size).pop_back();
      _curr_cache_entries--;

      return out;
    }
    return std::nullopt;
  }

  bool full() const { return _curr_cache_entries >= _max_cache_entries; }

 private:
  size_t _min_size_bytes = 20 * 1024 * 1024;
  size_t _max_cache_entries = 10;

  std::unordered_map<size_t, std::vector<CacheEntry>> _cache;
  size_t _curr_cache_entries = 0;
};

static AllocationCache allocation_cache;

DefaultMemoryHandle::DefaultMemoryHandle(size_t nbytes) {
  if (auto entry = allocation_cache.find(nbytes)) {
    _data = std::move(entry->first);
    _nbytes = entry->second;
  } else {
    _data = std::unique_ptr<uint8_t[]>(new uint8_t[nbytes]);
    _nbytes = nbytes;
  }
}

DefaultMemoryHandle::~DefaultMemoryHandle() {
  allocation_cache.insert(std::move(_data), _nbytes);
}

}  // namespace thirdai::smx