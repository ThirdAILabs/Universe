#pragma once

#include <rocksdb/db.h>
#include <search/src/neural_db/Chunk.h>
#include <search/src/neural_db/on_disk/Utils.h>
#include <unordered_set>

namespace thirdai::search::ndb {

class QueryToChunks {
 public:
  QueryToChunks(rocksdb::DB* db, rocksdb::ColumnFamilyHandle* column)
      : _db(db), _column(column) {}

  void addQueries(TxnPtr& txn, ChunkId start_id,
                  const std::vector<std::vector<ChunkId>>& chunk_ids);

  std::vector<ChunkId> getChunks(ChunkId query_id);

  void deleteChunks(TxnPtr& txn, const std::unordered_set<ChunkId>& chunks);

  ~QueryToChunks() { _db->DestroyColumnFamilyHandle(_column); }

 private:
  rocksdb::DB* _db;
  rocksdb::ColumnFamilyHandle* _column;
};

}  // namespace thirdai::search::ndb