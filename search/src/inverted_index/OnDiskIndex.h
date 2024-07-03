#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/DbAdapter.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <unordered_map>

namespace thirdai::search {

using DocId = uint64_t;
using DocScore = std::pair<DocId, float>;

using HashedToken = uint32_t;

class OnDiskIndex {
 public:
  explicit OnDiskIndex(const std::string& db_name, 
                       const DBAdapterConfig &db_adapter_config = DBAdapterConfig(),
                       const IndexConfig& config = IndexConfig());

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs);

  std::vector<DocScore> query(const std::string& query, uint32_t k) const;

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

  std::vector<HashedToken> tokenize(const std::string& text) const;

  std::unique_ptr<DbAdapter> _db;

  // Query variables
  uint64_t _max_docs_to_score;
  float _max_token_occurrence_frac;
  float _k1;
  float _b;

  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search