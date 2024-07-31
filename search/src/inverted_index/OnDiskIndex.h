#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <unordered_map>

namespace thirdai::search {

struct DocCount {
  DocCount(DocId doc_id, uint32_t count) : doc_id(doc_id), count(count) {}

  DocId doc_id : 40;
  uint32_t count : 24;
};

using HashedToken = uint32_t;

class OnDiskIndex final : public Retriever {
 public:
  explicit OnDiskIndex(const std::string& save_path,
                       const IndexConfig& config = IndexConfig());

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs) final;

  void update(const std::vector<DocId>& ids,
              const std::vector<std::string>& extra_tokens) final;

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize) const final;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize) const final;

  void remove(const std::vector<DocId>& id_list) final;

  size_t size() const final { return getNDocs(); }

  void save(const std::string& new_save_path) const final;

  static std::shared_ptr<OnDiskIndex> load(const std::string& save_path,
                                           bool read_only);

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "on-disk"; }

  void prune() final;

  ~OnDiskIndex() final;

 private:
  std::pair<std::vector<uint32_t>,
            std::vector<std::unordered_map<HashedToken, uint32_t>>>
  countTokenOccurences(const std::vector<std::string>& docs) const;

  std::unordered_map<DocId, float> scoreDocuments(
      const std::string& query) const;

  std::vector<HashedToken> tokenize(const std::string& text) const;

  void storeDocLens(const std::vector<DocId>& ids,
                    const std::vector<uint32_t>& doc_lens);

  void updateTokenToDocs(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_new_docs);

  void incrementDocLens(const std::vector<DocId>& ids,
                        const std::vector<uint32_t>& doc_len_increments);

  void incrementDocTokenCounts(
      const std::unordered_map<HashedToken, std::vector<DocCount>>&
          token_to_doc_updates);

  int64_t getDocLen(DocId doc_id) const;

  int64_t getNDocs() const;

  int64_t getSumDocLens() const;

  void updateNDocs(int64_t n_new_docs);

  void updateSumDocLens(int64_t sum_new_doc_lens);

  rocksdb::TransactionDB* _db;
  rocksdb::ColumnFamilyHandle* _default;
  rocksdb::ColumnFamilyHandle* _counters;
  rocksdb::ColumnFamilyHandle* _token_to_docs;

  std::string _save_path;

  // Query variables
  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;

  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search