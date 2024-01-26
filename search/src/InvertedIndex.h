#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

using DocId = uint64_t;
using Token = std::string;
using Tokens = std::vector<Token>;

using DocScore = std::pair<DocId, float>;

class InvertedIndex {
 public:
  static constexpr float DEFAULT_K = 1.2;
  static constexpr float DEFAULT_B = 0.75;

  explicit InvertedIndex(float k = DEFAULT_K, float b = DEFAULT_B)
      : _k(k), _b(b) {}

  void index(const std::vector<std::pair<DocId, Tokens>>& documents);

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<Tokens>& queries, uint32_t k) const;

  std::vector<DocScore> query(const Tokens& query, uint32_t k) const;

 private:
  void computeIdfs();

  inline float bm25(float idf, uint32_t freq, uint64_t doc_len) const {
    float num = freq * (_k + 1);
    float denom = freq + _k * (1 - _b + _b * doc_len / _avg_doc_length);
    return idf * num / denom;
  }

  using FreqInfo = std::pair<DocId, uint32_t>;

  std::unordered_map<Token, std::vector<FreqInfo>> _token_to_docs;
  std::unordered_map<Token, float> _token_to_idf;
  std::unordered_map<DocId, uint64_t> _doc_lengths;

  uint64_t _sum_doc_lens = 0;
  float _avg_doc_length;
  float _k, _b;
};

}  // namespace thirdai::search