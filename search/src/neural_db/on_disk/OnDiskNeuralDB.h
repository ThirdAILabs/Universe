#pragma once

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/neural_db/NeuralDB.h>
#include <search/src/neural_db/TextProcessor.h>
#include <search/src/neural_db/on_disk/ChunkCountView.h>
#include <search/src/neural_db/on_disk/ChunkDataColumn.h>
#include <search/src/neural_db/on_disk/InvertedIndex.h>
#include <search/src/neural_db/on_disk/QueryToChunks.h>
#include <search/src/neural_db/on_disk/Utils.h>
#include <memory>
#include <unordered_map>

namespace thirdai::search::ndb {

class OnDiskNeuralDB final : public NeuralDB {
 public:
  explicit OnDiskNeuralDB(const std::string& save_path,
                          const IndexConfig& config = IndexConfig(),
                          bool read_only = false);

  void insert(const std::string& document,
              const std::vector<std::string>& chunks,
              const std::vector<MetadataMap>& metadata,
              const std::optional<std::string>& doc_id) final;

  std::vector<std::pair<Chunk, float>> query(const std::string& query,
                                             uint32_t top_k) final;

  std::vector<std::pair<Chunk, float>> rank(const std::string& query,
                                            const QueryConstraints& constraints,
                                            uint32_t top_k) final;

  void finetune(const std::vector<std::vector<ChunkId>>& chunk_ids,
                const std::vector<std::string>& queries) final;

  void deleteDoc(const DocId& doc, uint32_t version) final;

  void prune() final;

  ~OnDiskNeuralDB() final;

 private:
  TxnPtr newTxn();

  std::unordered_map<ChunkId, float> candidateSet(const std::string& query);

  uint32_t getDocVersion(TxnPtr& txn, const std::string& doc_id);

  struct DocChunkRange {
    ChunkId start;
    ChunkId end;
  };

  DocChunkRange deleteDocChunkRange(TxnPtr& txn, const DocId& doc_id,
                                    uint32_t version);

  std::string _save_path;

  rocksdb::TransactionDB* _transaction_db;
  rocksdb::DB* _db;

  rocksdb::ColumnFamilyHandle* _default;

  std::unique_ptr<InvertedIndex> _chunk_index;

  std::unique_ptr<ChunkDataColumn<ChunkData>> _chunk_data;
  std::unique_ptr<ChunkDataColumn<MetadataMap>> _chunk_metadata;

  rocksdb::ColumnFamilyHandle* _doc_chunks;
  rocksdb::ColumnFamilyHandle* _doc_version;

  std::unique_ptr<InvertedIndex> _query_index;
  std::unique_ptr<QueryToChunks> _query_to_chunks;

  TextProcessor _text_processor;
};

}  // namespace thirdai::search::ndb