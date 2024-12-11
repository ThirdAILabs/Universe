#pragma once

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/neural_db/NeuralDb.h>
#include <search/src/neural_db/TextProcessor.h>
#include <search/src/neural_db/on_disk/ChunkCountView.h>
#include <unordered_map>

namespace thirdai::search::ndb {

using TxnPtr = std::unique_ptr<rocksdb::Transaction>;

class OnDiskNeuralDB final : public NeuralDB {
 public:
  explicit OnDiskNeuralDB(const std::string& save_path,
                          const IndexConfig& config = IndexConfig(),
                          bool read_only = false);

  void insert(const std::string& document,
              const std::optional<std::string>& doc_id,
              const std::vector<std::string>& chunks,
              const std::vector<MetadataMap>& metadata) final;

  std::vector<std::pair<Chunk, float>> query(const std::string& query,
                                             uint32_t top_k) final;

  std::vector<std::pair<Chunk, float>> rank(
      const std::string& query, uint32_t top_k,
      const QueryConstraints& constraints) final;

  void deleteDoc(const DocId& doc, uint32_t version) final;

  void prune() final;

  ~OnDiskNeuralDB() final;

 private:
  TxnPtr newTxn();

  ChunkId reserveChunkIds(ChunkId n_ids);

  uint32_t getDocVersion(TxnPtr& txn, const std::string& doc_id);

  void incrementCounter(TxnPtr& txn, const std::string& key, int64_t value);

  int64_t getCounter(const rocksdb::Slice& key);

  std::vector<rocksdb::PinnableSlice> mapTokensToChunks(
      const std::vector<HashedToken>& query_tokens);

  std::vector<std::pair<size_t, float>> rankByIdf(
      const std::vector<ChunkCountView>& token_to_chunks,
      int64_t n_chunks) const;

  std::unordered_map<ChunkId, float> candidateSet(const std::string& query);

  template <typename T>
  std::vector<std::optional<T>> loadChunkField(
      rocksdb::ColumnFamilyHandle* column,
      const std::vector<ChunkId>& chunk_ids);

  struct DocChunkRange {
    ChunkId start;
    ChunkId end;
  };

  DocChunkRange deleteDocChunkRange(TxnPtr& txn, const DocId& doc_id,
                                    uint32_t version);

  void removeChunksFromIndex(TxnPtr& txn, ChunkId start, ChunkId end);

  int64_t deleteChunkLens(TxnPtr& txn, const std::vector<ChunkId>& chunk_ids);

  static void deleteChunkField(TxnPtr& txn, rocksdb::ColumnFamilyHandle* column,
                               const std::vector<ChunkId>& chunk_ids);

  std::string _save_path;

  rocksdb::TransactionDB* _transaction_db;
  rocksdb::DB* _db;

  rocksdb::ColumnFamilyHandle* _default;
  rocksdb::ColumnFamilyHandle* _chunk_counters;
  rocksdb::ColumnFamilyHandle* _chunk_data;
  rocksdb::ColumnFamilyHandle* _chunk_metadata;
  rocksdb::ColumnFamilyHandle* _chunk_token_index;
  rocksdb::ColumnFamilyHandle* _doc_chunks;
  rocksdb::ColumnFamilyHandle* _doc_version;

  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;

  TextProcessor _text_processor;
};

}  // namespace thirdai::search::ndb