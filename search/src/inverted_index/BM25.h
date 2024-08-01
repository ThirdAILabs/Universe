#pragma once

#include <cmath>

namespace thirdai::search {

inline float idf(size_t n_docs, size_t docs_w_token) {
  const float num = n_docs - docs_w_token + 0.5;
  const float denom = docs_w_token + 0.5;
  // This is technically different from the BM25 definition, the added 1 is to
  // ensure that this does not yield a negative value. This trick is how
  // apache lucene solves the problem.
  return std::log(1.0 + num / denom);
}

inline float bm25(float idf, uint32_t cnt_in_doc, uint64_t doc_len,
                  float avg_doc_len, uint64_t query_len, float k1, float b) {
  const float num = cnt_in_doc * (k1 + 1);
  const float denom = cnt_in_doc + k1 * (1 - b + b * doc_len / avg_doc_len);
  return idf * num / (denom * query_len);
}

}  // namespace thirdai::search