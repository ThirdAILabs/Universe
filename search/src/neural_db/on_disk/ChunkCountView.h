#pragma once

#include <rocksdb/slice.h>
#include <search/src/neural_db/Chunk.h>

namespace thirdai::search::ndb {

static const ChunkCount PRUNED = ChunkCount{0xffffffffffffffff, 0xffffffff};

class ChunkCountView {
 public:
  explicit ChunkCountView(const std::vector<ChunkCount>& counts)
      : _ptr(counts.data()), _size(counts.size()) {}

  ChunkCountView(const char* data, size_t size)
      : _ptr(reinterpret_cast<const ChunkCount*>(data)),
        _size(size / sizeof(ChunkCount)) {}

  explicit ChunkCountView(const rocksdb::Slice& counts)
      : ChunkCountView(counts.data(), counts.size()) {}

  size_t size() const { return _size; }

  const auto* begin() const { return _ptr; }

  const auto* end() const { return _ptr + _size; }

  bool isPruned() const {
    return _size == 1 && _ptr->chunk_id == PRUNED.chunk_id &&
           _ptr->count == PRUNED.count;
  }

  rocksdb::Slice slice() const {
    return rocksdb::Slice(reinterpret_cast<const char*>(_ptr),
                          _size * sizeof(ChunkCount));
  }

 private:
  const ChunkCount* _ptr;
  size_t _size;
};

}  // namespace thirdai::search::ndb