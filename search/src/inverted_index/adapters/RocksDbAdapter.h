#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/DbAdapter.h>
#include <search/src/inverted_index/Retriever.h>
#include <unordered_set>

namespace thirdai::search {

class RocksDbAdapter final : public DbAdapter {
 public:
  explicit RocksDbAdapter(const std::string& save_path);

  void storeDocLens(const std::vector<DocId>& ids,
                    const std::vector<uint32_t>& doc_lens) final;

  void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs) final;

  std::vector<std::vector<DocCount>> lookupDocs(
      const std::vector<HashedToken>& query_tokens) const final;

  void prune(uint64_t max_docs_with_token) final;

  void removeDocs(const std::unordered_set<DocId>& docs) final;

  uint32_t getDocLen(DocId doc_id) const final;

  uint64_t getNDocs() const final;

  uint64_t getSumDocLens() const final;

  void save(const std::string& save_path) const final;

  std::string type() const final { return "rocksdb"; }

  ~RocksDbAdapter();

 private:
  void updateNDocs(uint64_t n_new_docs);

  void updateSumDocLens(uint64_t sum_new_doc_lens);

  rocksdb::TransactionDB* _db;
  rocksdb::ColumnFamilyHandle* _default;
  rocksdb::ColumnFamilyHandle* _counters;
  rocksdb::ColumnFamilyHandle* _token_to_docs;

  std::string _save_path;
};

}  // namespace thirdai::search