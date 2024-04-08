#include "FinetunableRetriever.h"
#include <memory>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::search {

FinetunableRetriever::FinetunableRetriever(float lambda, uint32_t min_top_docs,
                                           uint32_t top_queries)
    : _doc_index(std::make_shared<InvertedIndex>()),
      _query_index(std::make_shared<InvertedIndex>()),
      _lambda(lambda),
      _min_top_docs(min_top_docs),
      _top_queries(top_queries) {}

void FinetunableRetriever::index(const std::vector<DocId>& ids,
                                 const std::vector<std::string>& docs) {
  _doc_index->index(ids, docs);
}

void FinetunableRetriever::finetune(
    const std::vector<std::vector<DocId>>& doc_ids,
    const std::vector<std::string>& queries) {
  std::vector<QueryId> query_ids(doc_ids.size());
  std::iota(query_ids.begin(), query_ids.end(), _next_query_id);

  for (size_t i = 0; i < query_ids.size(); i++) {
    _query_to_docs[query_ids[i]] = doc_ids[i];
  }

  _query_index->index(query_ids, queries);

  _next_query_id += query_ids.size();
}

std::vector<DocScore> FinetunableRetriever::query(const std::string& query,
                                                  uint32_t k) const {
  auto top_docs = _doc_index->query(query, std::max(_min_top_docs, k));
  auto top_queries = _query_index->query(query, _top_queries);

  std::unordered_map<DocId, float> top_scores;
  for (const auto& [doc, score] : top_docs) {
    top_scores[doc] += _lambda * score;
  }

  for (const auto& [query, score] : top_queries) {
    for (DocId doc : _query_to_docs.at(query)) {
      top_scores[doc] += (1 - _lambda) * score;
    }
  }

  return InvertedIndex::topk(top_scores, k);
}

std::vector<std::vector<DocScore>> FinetunableRetriever::queryBatch(
    const std::vector<std::string>& queries, uint32_t k) const {
  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = query(queries[i], k);
  }

  return scores;
}

std::vector<DocScore> FinetunableRetriever::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k) const {
  auto top_docs =
      _doc_index->rank(query, candidates, std::max(_min_top_docs, k));
  auto top_queries = _query_index->query(query, _top_queries);

  std::unordered_map<DocId, float> top_scores;
  for (const auto& [doc, score] : top_docs) {
    top_scores[doc] += _lambda * score;
  }

  for (const auto& [query, score] : top_queries) {
    for (DocId doc : _query_to_docs.at(query)) {
      if (candidates.count(doc)) {
        top_scores[doc] += (1 - _lambda) * score;
      }
    }
  }

  return InvertedIndex::topk(top_scores, k);
}

std::vector<std::vector<DocScore>> FinetunableRetriever::rankBatch(
    const std::vector<std::string>& queries,
    const std::vector<std::unordered_set<DocId>>& candidates,
    uint32_t k) const {
  if (queries.size() != candidates.size()) {
    throw std::invalid_argument(
        "Number of queries must match number of candidate sets for ranking.");
  }

  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, candidates, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = rank(queries[i], candidates[i], k);
  }

  return scores;
}

}  // namespace thirdai::search