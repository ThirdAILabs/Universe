#pragma once

#include <rocksdb/slice.h>
#include <search/src/neural_db/Chunk.h>

namespace thirdai::search::ndb {

template <typename T>
class DataView {
 public:
  explicit DataView(const std::vector<T>& data)
      : _ptr(data.data()), _size(data.size()) {}

  DataView(const char* data, size_t size)
      : _ptr(reinterpret_cast<const T*>(data)), _size(size / sizeof(T)) {}

  explicit DataView(const rocksdb::Slice& counts)
      : DataView(counts.data(), counts.size()) {}

  size_t size() const { return _size; }

  const T* begin() const { return _ptr; }

  const T* end() const { return _ptr + _size; }

  rocksdb::Slice slice() const {
    return rocksdb::Slice(reinterpret_cast<const char*>(_ptr),
                          _size * sizeof(T));
  }

  std::vector<T> toVector() const { return {_ptr, _ptr + _size}; }

 protected:
  const T* _ptr;
  size_t _size;
};

static const ChunkCount PRUNED = ChunkCount{0xffffffffffffffff, 0xffffffff};

class ChunkCountView final : public DataView<ChunkCount> {
 public:
  using DataView<ChunkCount>::DataView;

  bool isPruned() const {
    return _size == 1 && _ptr->chunk_id == PRUNED.chunk_id &&
           _ptr->count == PRUNED.count;
  }
};

}  // namespace thirdai::search::ndb