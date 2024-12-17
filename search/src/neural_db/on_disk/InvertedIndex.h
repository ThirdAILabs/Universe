#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/neural_db/TextProcessor.h>
#include <search/src/neural_db/on_disk/DataView.h>
#include <search/src/neural_db/on_disk/Utils.h>
#include <unordered_set>

namespace thirdai::search::ndb {

class InvertedIndex {
 public:
  InvertedIndex(rocksdb::DB* db, rocksdb::ColumnFamilyHandle* counters,
                rocksdb::ColumnFamilyHandle* token_index,
                const IndexConfig& config, bool read_only);

  void insert(TxnPtr& txn, ChunkId start_id, const BatchTokens& token_counts,
              const std::vector<uint32_t>& lens);

  ChunkId reserveChunkIds(TxnPtr& txn, ChunkId n_ids);

  std::unordered_map<ChunkId, float> candidateSet(
      const std::vector<std::string>& query_tokens,
      int64_t min_chunks_for_idf = 0);

  void deleteChunks(TxnPtr& txn, const std::unordered_set<ChunkId>& chunk_ids);

  void prune(TxnPtr& txn);

  size_t size();

  ~InvertedIndex() {
    _db->DestroyColumnFamilyHandle(_counters);
    _db->DestroyColumnFamilyHandle(_token_index);
  }

 private:
  std::vector<rocksdb::PinnableSlice> mapTokensToChunks(
      const std::vector<std::string>& query_tokens);

  std::vector<std::pair<size_t, float>> rankByIdf(
      const std::vector<ChunkCountView>& token_to_chunks,
      int64_t n_chunks) const;

  void initCounter(const std::string& key, int64_t value);

  void incrementCounter(TxnPtr& txn, const std::string& key, int64_t value);

  int64_t getCounter(const rocksdb::Slice& key);

  rocksdb::DB* _db;

  rocksdb::ColumnFamilyHandle* _counters;
  rocksdb::ColumnFamilyHandle* _token_index;

  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;
};

}  // namespace thirdai::search::ndb