#include "ShardedIndex.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>

namespace thirdai::search {

void ShardedIndex::index(const std::vector<DocId>& ids,
                         const std::vector<std::string>& docs) {
  if (ids.size() != docs.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  size_t offset = 0;
  while (offset < ids.size()) {
    if (_indexes.empty() || _indexes.back()->size() >= _max_shard_size) {
      _indexes.push_back(std::make_shared<InvertedIndex>());
    }

    size_t chunksize = std::min(_max_shard_size - _indexes.back()->size(),
                                ids.size() - offset);

    _indexes.back()->index(
        {ids.begin() + offset, ids.begin() + offset + chunksize},
        {docs.begin() + offset, docs.begin() + offset + chunksize});

    offset += chunksize;
  }
}

std::vector<DocScore> topk(
    const std::vector<std::vector<DocScore>>& shard_scores, uint32_t k) {
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);

  const HighestScore<DocId> cmp;
  for (const auto& scores : shard_scores) {
    for (const auto& [doc, score] : scores) {
      if (top_scores.size() < k || top_scores.front().second < score) {
        top_scores.emplace_back(doc, score);
        std::push_heap(top_scores.begin(), top_scores.end(), cmp);
      }

      if (top_scores.size() > k) {
        std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
        top_scores.pop_back();
      }
    }
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

std::vector<DocScore> ShardedIndex::query(const std::string& query, uint32_t k,
                                          bool parallelize) const {
  std::vector<std::vector<DocScore>> shard_scores(_indexes.size());

#pragma omp parallel for default(none) \
    shared(shard_scores, query, k) if (_indexes.size() > 1 && parallelize)
  for (size_t i = 0; i < _indexes.size(); i++) {
    shard_scores[i] = _indexes[i]->query(query, k);
  }

  return topk(shard_scores, k);
}

std::vector<DocScore> ShardedIndex::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k, bool parallelize) const {
  std::vector<std::vector<DocScore>> shard_scores(_indexes.size());

#pragma omp parallel for default(none)      \
    shared(shard_scores, query, candidates, \
           k) if (_indexes.size() > 1 && parallelize)
  for (size_t i = 0; i < _indexes.size(); i++) {
    shard_scores[i] = _indexes[i]->rank(query, candidates, k);
  }

  return topk(shard_scores, k);
}

void ShardedIndex::remove(const std::vector<DocId>& ids) {
  for (const auto& index : _indexes) {
    index->remove(ids);
  }
}

size_t ShardedIndex::size() const {
  size_t size = 0;
  for (const auto& index : _indexes) {
    size += index->size();
  }
  return size;
}

ar::ConstArchivePtr ShardedIndex::toArchive() const {
  auto map = ar::Map::make();

  auto indexes = ar::List::make();
  for (const auto& index : _indexes) {
    indexes->append(index->toArchive());
  }

  map->set("indexes", indexes);
  map->set("max_shard_size", ar::u64(_max_shard_size));

  return map;
}

ShardedIndex::ShardedIndex(const ar::Archive& archive)
    : _max_shard_size(archive.u64("max_shard_size")) {
  for (const auto& index_archive : archive.get("indexes")->list()) {
    _indexes.push_back(InvertedIndex::fromArchive(*index_archive));
  }
}

std::shared_ptr<ShardedIndex> ShardedIndex::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<ShardedIndex>(archive);
}

}  // namespace thirdai::search