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

    // TODO(Nicholas): Should this index creation be parallelized, currently the
    // index construction time is only a few seconds. If so, is it faster to
    // have a critical section around the following lines or have the
    // frequencies computed foreach doc in parallel and then aggregate everyting
    // serially at the end.
    for (const auto& [token, freq] : freqs) {
      _token_to_docs[token].emplace_back(doc_id, freq);
    }
    _doc_lengths[doc_id] = tokens.size();
    _sum_doc_lens += tokens.size();
  }

  computeIdfs();
  _avg_doc_length = static_cast<float>(_sum_doc_lens) / _doc_lengths.size();
}

inline float idf(size_t n_docs, size_t docs_w_token) {
  const float num = n_docs - docs_w_token + 0.5;
  const float denom = docs_w_token + 0.5;
  return std::log(num / denom);
}

void InvertedIndex::computeIdfs() {
  const size_t n_docs = _doc_lengths.size();

  // We can calculate the idf of a hypothetical token that occured in the
  // specified fraction of the documents. We know that any idf less than this
  // corresponds to a token that occurs in more than that fraction of docs. An
  // alternative idea would be to throw away the x% most common tokens (lowest
  // idf).
  const size_t max_docs_with_token = n_docs * _idf_cutoff_frac;
  const float idf_cutoff = idf(n_docs, max_docs_with_token);

  _token_to_idf.clear();
  for (const auto& [token, docs] : _token_to_docs) {
    const size_t docs_w_token = docs.size();
    const float idf_score = idf(n_docs, docs_w_token);
    if (idf_score >= idf_cutoff) {
      _token_to_idf[token] = idf_score;
    }
  }
}

std::vector<std::vector<DocScore>> InvertedIndex::queryBatch(
    const std::vector<Tokens>& queries, uint32_t k) const {
  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = query(queries[i], k);
  }

  return scores;
}

struct HighestScore {
  bool operator()(const DocScore& a, const DocScore& b) const {
    return a.second > b.second;
  }
};

std::vector<DocScore> InvertedIndex::query(const Tokens& query,
                                           uint32_t k) const {
  std::unordered_map<DocId, float> doc_scores;

  for (const Token& token : query) {
    if (!_token_to_idf.count(token)) {
      continue;
    }
    const float token_idf = _token_to_idf.at(token);

    for (const auto& [doc_id, token_freq] : _token_to_docs.at(token)) {
      const uint64_t doc_len = _doc_lengths.at(doc_id);

      // Note: This bm25 score could be precomputed for each (token, doc) pair.
      // However it would mean that all scores would need to be recomputed when
      // more docs are added since the idf and avg_doc_len will change. So if we
      // do not need to support small incremental additions then it might make
      // sense to precompute these values.
      doc_scores[doc_id] += bm25(token_idf, token_freq, doc_len);
    }
  }

  // Using a heap like this is O(N log(K)) where N is the number of docs.
  // Sorting the entire list and taking the top K would be O(N log(N)).
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  const HighestScore cmp;

  for (const auto& [doc, score] : doc_scores) {
    if (top_scores.size() < k || top_scores.front().second < score) {
      top_scores.emplace_back(doc, score);
      std::push_heap(top_scores.begin(), top_scores.end(), cmp);
    }

    if (top_scores.size() > k) {
      std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
      top_scores.pop_back();
    }
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

}  // namespace thirdai::search