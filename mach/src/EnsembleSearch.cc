#include "EnsembleSearch.h"
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/ValueColumns.h>
#include <mach/src/MachRetriever.h>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace thirdai::mach {

bolt::TensorList EnsembleSearch::scoreBuckets(
    const std::vector<MachRetrieverPtr>& retrievers,
    std::vector<std::string> queries) {
  data::ColumnMap columns(
      {{retrievers[0]->_text_column,
        data::ValueColumn<std::string>::make(std::move(queries))}});

  bolt::TensorList scores(retrievers.size());

#pragma omp parallel for default(none) \
    shared(retrievers, columns, scores) if (queries.size() == 1)
  for (size_t i = 0; i < retrievers.size(); i++) {
    auto tensors = retrievers[i]->inputTensors(
        retrievers[i]->_text_transform->apply(columns, *retrievers[i]->_state));

    scores[i] = retrievers[i]->_model->forward(tensors, false).at(0);
  }

  return scores;
}

std::unordered_set<uint32_t> EnsembleSearch::aggregateCandidates(
    const std::vector<MachRetrieverPtr>& retrievers,
    const bolt::TensorList& scores, size_t index_in_batch) {
  std::vector<std::unordered_set<uint32_t>> candidates(scores.size());

#pragma omp parallel for default(none)                 \
    shared(retrievers, candidates, candidates, scores, \
           index_in_batch) if (scores[0]->batchSize() == 1)
  for (size_t ret = 0; ret < scores.size(); ret++) {
    auto top_buckets = scores[ret]
                           ->getVector(index_in_batch)
                           .topKNeurons(retrievers[ret]->_n_buckets_to_eval);

    const auto& index = retrievers[ret]->index();

    while (!top_buckets.empty()) {
      for (uint32_t id : index->getEntities(top_buckets.top().second)) {
        candidates[ret].insert(id);
      }
      top_buckets.pop();
    }
  }

  std::unordered_set<uint32_t> all_candidates;

  for (const auto& candidate_set : candidates) {
    all_candidates.insert(candidate_set.begin(), candidate_set.end());
  }

  return all_candidates;
}

void scoreCandidates(const std::vector<MachRetrieverPtr>& retrievers,
                     IdScores& candidates, const bolt::TensorList& scores,
                     size_t index_in_batch) {
  std::vector<IdScores> retriever_scores(retrievers.size());

#pragma omp parallel for default(none)                       \
    shared(retrievers, candidates, retriever_scores, scores, \
           index_in_batch) if (scores[0]->batchSize() == 1)
  for (size_t ret = 0; ret < retrievers.size(); ret++) {
    retriever_scores[ret] = candidates;

    const auto& index = retrievers[ret]->index();
    const BoltVector& vec = scores[ret]->getVector(index_in_batch);

    for (auto& [id, score] : retriever_scores[ret]) {
      for (uint32_t hash : index->getHashes(id)) {
        score += vec.activations[hash];
      }
    }
  }

  for (const auto& ret_scores : retriever_scores) {
    for (size_t i = 0; i < candidates.size(); i++) {
      candidates[i].second += ret_scores[i].second;
    }
  }
}

struct BestScore {
  bool operator()(const std::pair<uint32_t, float>& a,
                  const std::pair<uint32_t, float>& b) {
    return a.second > b.second;
  }
};

std::vector<IdScores> EnsembleSearch::searchEnsemble(
    const std::vector<MachRetrieverPtr>& retrievers,
    const std::vector<std::string>& queries, uint32_t topk) {
  auto scores = scoreBuckets(retrievers, queries);

  std::vector<IdScores> output(queries.size());

#pragma omp parallel for default(none) \
    shared(retrievers, queries, topk, scores, output) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    auto candidates = aggregateCandidates(retrievers, scores, i);

    IdScores candidate_scores;
    candidate_scores.reserve(candidates.size());
    for (uint32_t candidate : candidates) {
      candidate_scores.emplace_back(candidate, 0);
    }

    scoreCandidates(retrievers, candidate_scores, scores, i);

    std::sort(candidate_scores.begin(), candidate_scores.end(), BestScore{});
    if (candidate_scores.size() > topk) {
      candidate_scores.resize(topk);
    }

    output[i] = std::move(candidate_scores);
  }

  return output;
}

std::vector<IdScores> EnsembleSearch::rankEnsemble(
    const std::vector<MachRetrieverPtr>& retrievers,
    const std::vector<std::string>& queries,
    const std::vector<std::unordered_set<uint32_t>>& candidates,
    uint32_t topk) {
  if (queries.size() != candidates.size()) {
    throw std::invalid_argument(
        "Number of candidate sets and queries must match.");
  }
  auto scores = scoreBuckets(retrievers, queries);

  std::vector<IdScores> output(queries.size());

#pragma omp parallel for default(none)                    \
    shared(retrievers, queries, candidates, topk, scores, \
           output) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    IdScores candidate_scores;
    candidate_scores.reserve(candidates[i].size());
    for (uint32_t candidate : candidates[i]) {
      candidate_scores.emplace_back(candidate, 0);
    }

    scoreCandidates(retrievers, candidate_scores, scores, i);

    std::sort(candidate_scores.begin(), candidate_scores.end(), BestScore{});
    if (candidate_scores.size() > topk) {
      candidate_scores.resize(topk);
    }

    output[i] = std::move(candidate_scores);
  }

  return output;
}

}  // namespace thirdai::mach