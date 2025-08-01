#include "QueryToChunks.h"
#include <rocksdb/iterator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <search/src/neural_db/Chunk.h>
#include <search/src/neural_db/on_disk/DataView.h>
#include <search/src/neural_db/on_disk/RocksDBError.h>
#include <stdexcept>

namespace thirdai::search::ndb {

rocksdb::Slice vecToSlice(const std::vector<ChunkId>& vec) {
  return rocksdb::Slice(reinterpret_cast<const char*>(vec.data()),
                        vec.size() * sizeof(ChunkId));
}

void QueryToChunks::addQueries(
    TxnPtr& txn, ChunkId start_id,
    const std::vector<std::vector<ChunkId>>& chunk_ids) {
  for (size_t i = 0; i < chunk_ids.size(); i++) {
    ChunkId query_id = start_id + i;

    auto status = txn->Merge(_column, asSlice<ChunkId>(&query_id),
                             vecToSlice(chunk_ids[i]));
    if (!status.ok()) {
      throw RocksdbError(status, "logging finetuning");
    }
  }
}

std::vector<ChunkId> QueryToChunks::getChunks(ChunkId query_id) {
  const auto key = asSlice<ChunkId>(&query_id);

  std::string value;
  auto status = _db->Get(rocksdb::ReadOptions(), _column, key, &value);
  if (!status.ok()) {
    throw RocksdbError(status, "querying ndb");
  }

  DataView<ChunkId> view(value);
  return view.toVector();
}

void QueryToChunks::deleteChunks(TxnPtr& txn,
                                 const std::unordered_set<ChunkId>& chunks) {
  auto iter = std::unique_ptr<rocksdb::Iterator>(
      txn->GetIterator(rocksdb::ReadOptions(), _column));

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    const auto value = iter->value();

    DataView<ChunkId> view(value);

    std::vector<ChunkId> new_value;
    new_value.reserve(view.size());
    for (const auto& id : view) {
      if (!chunks.count(id)) {
        new_value.push_back(id);
      }
    }

    if (new_value.size() != view.size()) {
      auto put_status = txn->Put(_column, iter->key(), vecToSlice(new_value));
      if (!put_status.ok()) {
        throw RocksdbError(put_status, "removing doc chunk refs");
      }
    }
  }
}

}  // namespace thirdai::search::ndb