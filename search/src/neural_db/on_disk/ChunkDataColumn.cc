#include "ChunkDataColumn.h"
#include <search/src/neural_db/Constraints.h>
#include <search/src/neural_db/on_disk/RocksDBError.h>
#include <search/src/neural_db/on_disk/Serialization.h>
#include <unordered_map>

namespace thirdai::search::ndb {

template <typename T>
void ChunkDataColumn<T>::write(TxnPtr& txn, ChunkId start_id,
                               const std::vector<T>& data) {
  for (size_t i = 0; i < data.size(); i++) {
    const ChunkId chunk_id = start_id + i;
    const auto chunk_key = asSlice<ChunkId>(&chunk_id);

    std::string chunk_data = serialize<T>(data[i]);

    auto status = txn->Put(_column, chunk_key, chunk_data);
    if (!status.ok()) {
      throw RocksdbError(status, "storing chunk metadata");
    }
  }
}

template <typename T>
std::vector<std::optional<T>> ChunkDataColumn<T>::get(
    const std::vector<ChunkId>& chunk_ids) {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(chunk_ids.size());

  for (const auto& id : chunk_ids) {
    keys.emplace_back(asSlice<ChunkId>(&id));
  }

  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), _column, keys.size(), keys.data(),
                values.data(), statuses.data());

  std::vector<std::optional<T>> result;
  result.reserve(chunk_ids.size());

  for (size_t i = 0; i < chunk_ids.size(); i++) {
    if (statuses[i].ok()) {
      result.emplace_back(deserialize<T>(values[i]));
    } else if (statuses[i].IsNotFound()) {
      // This could happen if chunks are deleted between the first and second
      // phase of a query.
      result.emplace_back(std::nullopt);
    } else {
      throw RocksdbError(statuses[i], "retrieving chunk metadata");
    }
  }

  return result;
}

template <typename T>
void ChunkDataColumn<T>::remove(TxnPtr& txn,
                                const std::unordered_set<ChunkId>& chunk_ids) {
  for (const auto& chunk_id : chunk_ids) {
    auto status = txn->Delete(_column, asSlice<ChunkId>(&chunk_id));
    if (!status.ok()) {
      throw RocksdbError(status, "removing chunk metadata");
    }
  }
}

template class ChunkDataColumn<MetadataMap>;
template class ChunkDataColumn<ChunkData>;

}  // namespace thirdai::search::ndb