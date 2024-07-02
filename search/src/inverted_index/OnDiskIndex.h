#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <unordered_map>

namespace thirdai::search {

using HashedToken = uint32_t;

class OnDiskIndex final : public Retriever {
 public:
  explicit OnDiskIndex(const std::string& save_path,
                       const IndexConfig& config = IndexConfig());

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs) final;

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize) const final;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize) const final;

  void remove(const std::vector<DocId>& ids) final;

  size_t size() const final { return getNDocs(); }

  void save(const std::string& new_save_path) const final;

  static std::shared_ptr<OnDiskIndex> load(const std::string& save_path);

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "on-disk"; }

  void prune() final;

  ~OnDiskIndex() final;

 private:
  static inline float idf(size_t n_docs, size_t docs_w_token) {
    const float num = n_docs - docs_w_token + 0.5;
    const float denom = docs_w_token + 0.5;
    // This is technically different from the BM25 definition, the added 1 is to
    // ensure that this does not yield a negative value. This trick is how
    // apache lucene solves the problem.
    return std::log(1.0 + num / denom);
  }

  inline float bm25(float idf, uint32_t cnt_in_doc, uint64_t doc_len,
                    float avg_doc_len) const {
    const float num = cnt_in_doc * (_k1 + 1);
    const float denom =
        cnt_in_doc + _k1 * (1 - _b + _b * doc_len / avg_doc_len);
    return idf * num / denom;
  }

  std::pair<std::vector<uint32_t>,
            std::vector<std::unordered_map<HashedToken, uint32_t>>>
  countTokenOccurences(const std::vector<std::string>& docs) const;

  void storeDocLens(const std::vector<DocId>& ids,
                    const std::vector<uint32_t>& doc_lens,
                    bool check_for_existing);

  void updateTokenToDocs(
      const std::vector<DocId>& ids,
      const std::vector<std::unordered_map<HashedToken, uint32_t>>&
          token_counts);

  std::unordered_map<DocId, float> scoreDocuments(
      const std::string& query) const;

  uint32_t getDocLen(DocId doc_id) const;

  uint64_t getNDocs() const;

  void updateNDocs(uint64_t n_new_docs);

  uint64_t getSumDocLens() const;

  void updateSumDocLens(uint64_t sum_new_doc_lens);

  std::vector<HashedToken> tokenize(const std::string& text) const;

  std::string _save_path;

  rocksdb::TransactionDB* _db;
  rocksdb::ColumnFamilyHandle* _default;
  rocksdb::ColumnFamilyHandle* _counters;
  rocksdb::ColumnFamilyHandle* _token_to_docs;

  // Query variables
  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;

  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search