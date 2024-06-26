#pragma once

#include <rocksdb/db.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <shared_mutex>
#include <unordered_map>

namespace thirdai::search {

using DocId = uint64_t;
using DocScore = std::pair<DocId, float>;

class OnDiskIndex {
 public:
  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs);

  std::vector<DocScore> query(const std::string& query, uint32_t k);

  bool containsDoc(DocId doc_id) const;

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

  std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>>
  countTokenOccurences(const std::vector<std::string>& docs) const;

  uint32_t getDocLen(DocId doc_id);

  rocksdb::DB* _db;

  // Cached state: Can be refreshed from db on load
  std::shared_mutex _mutex;

  uint64_t _sum_doc_lens = 0;
  uint64_t _n_docs = 0;
  float _avg_doc_len;

  std::pair<uint64_t, float> getNDocsAndAvgLen();

  void updateNDocsAndAvgLen(uint64_t sum_new_doc_lens, uint64_t n_new_docs);

  // Query variables
  size_t _max_docs_to_score;
  float _idf_cutoff_frac;
  float _k1, _b;

  std::vector<uint32_t> tokenize(const std::string& text) const;

  TokenizerPtr _tokenizer;
};

}  // namespace thirdai::search