#pragma once

#include <rocksdb/db.h>
#include <search/src/neural_db/Chunk.h>
#include <search/src/neural_db/on_disk/Utils.h>
#include <unordered_set>

namespace thirdai::search::ndb {

template <typename T>
class ChunkDataColumn {
 public:
  ChunkDataColumn(rocksdb::DB* db, rocksdb::ColumnFamilyHandle* column)
      : _db(db), _column(column) {}

  void write(TxnPtr& txn, ChunkId start_id, const std::vector<T>& data);

  std::vector<std::optional<T>> get(const std::vector<ChunkId>& chunk_ids);

  void remove(TxnPtr& txn, const std::unordered_set<ChunkId>& chunk_ids);

  ~ChunkDataColumn() { _db->DestroyColumnFamilyHandle(_column); }

 private:
  rocksdb::DB* _db;
  rocksdb::ColumnFamilyHandle* _column;
};

}  // namespace thirdai::search::ndb