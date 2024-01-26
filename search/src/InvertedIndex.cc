#include "InvertedIndex.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

void InvertedIndex::index(
    const std::vector<std::pair<DocId, Tokens>>& documents) {
  for (const auto& [doc_id, tokens] : documents) {
    if (_doc_lengths.count(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    std::unordered_map<Token, uint32_t> freqs;
    for (const auto& token : tokens) {
      freqs[token]++;
    }

    // The rest of this would need to be in a critical section.
    // TODO(Nicholas): is it faster to have a critical section around the
    // following or have the frequencies computed foreach doc and then aggregate
    // everyting serially at the end.
    for (const auto& [token, freq] : freqs) {
      _token_to_docs[token].emplace_back(doc_id, freq);
    }
    _doc_lengths[doc_id] = tokens.size();
    _sum_doc_lens += tokens.size();
  }

  computeIdfs();
  _avg_doc_length = static_cast<float>(_sum_doc_lens) / _doc_lengths.size();
}

void InvertedIndex::computeIdfs() {
  _token_to_idf.clear();
  for (const auto& [token, docs] : _token_to_docs) {
    size_t docs_w_token = docs.size();

    float num = _doc_lengths.size() - docs_w_token + 0.5;
    float denom = docs_w_token + 0.5;

    _token_to_idf[token] = std::log(num / denom);
  }

  // TODO(Nicholas): Logic to prune elements with low idf.
}

std::vector<std::vector<DocScore>> InvertedIndex::query(
    const std::vector<Tokens>& queries, uint32_t k) const {
  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = query(queries[i], k);
  }

  return scores;
}

struct CompareScores {
  bool operator()(const DocScore& a, const DocScore& b) const {
    return a.second > b.second;
  }
};

std::vector<DocScore> InvertedIndex::query(const Tokens& query,
                                           uint32_t k) const {
  std::unordered_map<DocId, float> scores;

  for (const Token& token : query) {
    if (!_token_to_idf.count(token)) {
      continue;
    }
    float token_idf = _token_to_idf.at(token);

    for (const auto& [doc_id, token_freq] : _token_to_docs.at(token)) {
      uint64_t doc_len = _doc_lengths.at(doc_id);

      // Note: This bm25 could be precomputed for each (token, doc) pair.
      // However it would mean that all scores would need to be recomputed when
      // more docs are added since the idf and avg_doc_len will change. So if we
      // do not need to support small incremental additions then it might make
      // sense to precompute these values.
      scores[doc_id] += bm25(token_idf, token_freq, doc_len);
    }
  }

  std::vector<std::pair<DocId, float>> top_scores(scores.begin(), scores.end());

  CompareScores cmp;
  std::sort(top_scores.begin(), top_scores.end(), cmp);
  if (top_scores.size() > k) {
    top_scores.resize(k);
  }

  return top_scores;
}

}  // namespace thirdai::search