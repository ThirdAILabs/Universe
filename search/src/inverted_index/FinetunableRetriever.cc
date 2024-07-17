#include "FinetunableRetriever.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <search/src/inverted_index/OnDiskIndex.h>
#include <search/src/inverted_index/ShardedRetriever.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <search/src/inverted_index/Utils.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::search {

/**
 * Because of how bm25 is calculated, particularly the idf scores, the query
 * index does not work will with only a couple of finetuning samples, for
 * instance a single upvote/associate. Thus for a small number of finetuning
 * samples we concatenate the query to the documents it maps to, this boosts the
 * score for the document for that query maps to. The samples are still added to
 * the query index, just the query index isn't used until this threshold of
 * samples is reached.
 */
constexpr size_t QUERY_INDEX_THRESHOLD = 10;

namespace {

std::string docIndexPath(const std::string& save_path) {
  return (std::filesystem::path(save_path) / "primary").string();
}

std::string queryIndexPath(const std::string& save_path) {
  return (std::filesystem::path(save_path) / "secondary").string();
}

}  // namespace

FinetunableRetriever::FinetunableRetriever(
    float lambda, uint32_t min_top_docs, uint32_t top_queries,
    const IndexConfig& config, const std::optional<std::string>& save_path)
    : _lambda(lambda), _min_top_docs(min_top_docs), _top_queries(top_queries) {
  if (save_path) {
    createDirectory(*save_path);

    _doc_index =
        std::make_shared<ShardedRetriever>(config, docIndexPath(*save_path));
    _query_index =
        std::make_shared<OnDiskIndex>(queryIndexPath(*save_path), config);
  } else {
    _doc_index = std::make_shared<InvertedIndex>(config);
    _query_index = std::make_shared<InvertedIndex>(config);
  }

  _doc_index_type = _doc_index->type();
  _query_index_type = _query_index->type();
}

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
    for (DocId doc : doc_ids[i]) {
      _doc_to_queries[doc].push_back(query_ids[i]);
    }
  }

  _query_index->index(query_ids, queries);

  if (_query_index->size() < QUERY_INDEX_THRESHOLD) {
    std::vector<DocId> flattened_doc_ids;
    std::vector<std::string> flattened_queries;
    for (size_t i = 0; i < doc_ids.size(); i++) {
      const auto& ids = doc_ids[i];
      flattened_doc_ids.insert(flattened_doc_ids.end(), ids.begin(), ids.end());
      flattened_queries.insert(flattened_queries.end(), ids.size(), queries[i]);
    }

    _doc_index->update(flattened_doc_ids, flattened_queries);
  }

  _next_query_id += query_ids.size();
}

void FinetunableRetriever::associate(const std::vector<std::string>& sources,
                                     const std::vector<std::string>& targets,
                                     uint32_t strength) {
  auto top_docs = queryBatch(targets, /*k=*/strength);

  std::vector<std::vector<DocId>> ids_only;
  ids_only.reserve(top_docs.size());
  for (const auto& doc_scores : top_docs) {
    std::vector<DocId> ids;
    ids.reserve(doc_scores.size());
    for (auto id_score : doc_scores) {
      ids.push_back(id_score.first);
    }
    ids_only.push_back(ids);
  }

  finetune(ids_only, sources);
}

std::vector<DocScore> FinetunableRetriever::query(const std::string& query,
                                                  uint32_t k,
                                                  bool parallelize) const {
  if (_query_index->size() < QUERY_INDEX_THRESHOLD) {
    return _doc_index->query(query, k, parallelize);
  }

  auto top_docs =
      _doc_index->query(query, std::max(_min_top_docs, k), parallelize);
  auto top_queries = _query_index->query(query, _top_queries, parallelize);

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
    scores[i] = query(queries[i], k, /*parallelize=*/false);
  }

  return scores;
}

std::vector<DocScore> FinetunableRetriever::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k, bool parallelize) const {
  if (_query_index->size() < QUERY_INDEX_THRESHOLD) {
    return _doc_index->rank(query, candidates, k, parallelize);
  }

  auto top_docs = _doc_index->rank(query, candidates,
                                   std::max(_min_top_docs, k), parallelize);
  auto top_queries = _query_index->query(query, _top_queries, parallelize);

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
    scores[i] = rank(queries[i], candidates[i], k, /*parallelize=*/false);
  }

  return scores;
}

void FinetunableRetriever::remove(const std::vector<DocId>& ids) {
  _doc_index->remove(ids);

  std::vector<QueryId> irrelevant_queries;
  for (DocId doc : ids) {
    if (!_doc_to_queries.count(doc)) {
      continue;
    }
    for (QueryId query : _doc_to_queries.at(doc)) {
      auto& docs_for_query = _query_to_docs.at(query);
      auto loc = std::find(docs_for_query.begin(), docs_for_query.end(), doc);
      if (loc != docs_for_query.end()) {
        docs_for_query.erase(loc);
      }
      if (docs_for_query.empty()) {
        irrelevant_queries.push_back(query);
      }
    }
    _doc_to_queries.erase(doc);
  }

  _query_index->remove(irrelevant_queries);
  for (QueryId query : irrelevant_queries) {
    if (_query_to_docs.count(query)) {
      _query_to_docs.erase(query);
    }
  }
}

ar::ConstArchivePtr FinetunableRetriever::metadataToArchive() const {
  auto map = ar::Map::make();

  map->set("doc_index_type", ar::str(_doc_index_type));
  map->set("query_index_type", ar::str(_query_index_type));

  map->set("query_to_docs", ar::mapU64VecU64(_query_to_docs));
  map->set("doc_to_queries", ar::mapU64VecU64(_doc_to_queries));

  map->set("next_query_id", ar::u64(_next_query_id));

  map->set("lambda", ar::f32(_lambda));
  map->set("min_top_docs", ar::u64(_min_top_docs));
  map->set("top_queries", ar::u64(_top_queries));

  return map;
}

void FinetunableRetriever::metadataFromArchive(const ar::Archive& archive) {
  _doc_index_type = archive.str("doc_index_type");
  _query_index_type = archive.str("query_index_type");
  _query_to_docs = archive.getAs<ar::MapU64VecU64>("query_to_docs");
  _doc_to_queries = archive.getAs<ar::MapU64VecU64>("doc_to_queries");
  _next_query_id = archive.u64("next_query_id");
  _lambda = archive.f32("lambda");
  _min_top_docs = archive.u64("min_top_docs");
  _top_queries = archive.u64("top_queries");
}

namespace {

std::string metadataPath(const std::string& save_path) {
  return (std::filesystem::path(save_path) / "metadata").string();
}

}  // namespace

void FinetunableRetriever::save(const std::string& save_path) const {
  createDirectory(save_path);

  _doc_index->save(docIndexPath(save_path));
  _query_index->save(queryIndexPath(save_path));

  auto metadata = dataset::SafeFileIO::ofstream(metadataPath(save_path));
  ar::serialize(metadataToArchive(), metadata);
}

std::shared_ptr<Retriever> loadIndex(const std::string& type,
                                     const std::string& path, bool read_only) {
  if (type == InvertedIndex::typeName()) {
    return InvertedIndex::load(path);
  }
  if (type == OnDiskIndex::typeName()) {
    return OnDiskIndex::load(path, read_only);
  }
  if (type == ShardedRetriever::typeName()) {
    return ShardedRetriever::load(path, read_only);
  }
  throw std::invalid_argument("Invalid retriever type '" + type + "'.");
}

FinetunableRetriever::FinetunableRetriever(const std::string& save_path,
                                           bool read_only) {
  auto metadata = dataset::SafeFileIO::ifstream(metadataPath(save_path));
  metadataFromArchive(*ar::deserialize(metadata));

  _doc_index = loadIndex(_doc_index_type, docIndexPath(save_path), read_only);
  _query_index =
      loadIndex(_query_index_type, queryIndexPath(save_path), read_only);
}

std::shared_ptr<FinetunableRetriever> FinetunableRetriever::load(
    const std::string& save_path, bool read_only) {
  return std::shared_ptr<FinetunableRetriever>(
      new FinetunableRetriever(save_path, read_only));
}

void FinetunableRetriever::save_stream(std::ostream& ostream) const {
  (void)ostream;
  (void)this;  // Otherwise clang-tidy wants this to be static.
  throw std::invalid_argument(
      "Pickling is not supported for FinetunableRetriever.");
}

FinetunableRetriever::FinetunableRetriever(const ar::Archive& archive)
    : _doc_index(InvertedIndex::fromArchive(*archive.get("doc_index"))),
      _query_index(InvertedIndex::fromArchive(*archive.get("query_index"))),
      _query_to_docs(archive.getAs<ar::MapU64VecU64>("query_to_docs")),
      _doc_to_queries(archive.getAs<ar::MapU64VecU64>("doc_to_queries")),
      _next_query_id(archive.u64("next_query_id")),
      _lambda(archive.f32("lambda")),
      _min_top_docs(archive.u64("min_top_docs")),
      _top_queries(archive.u64("top_queries")) {}

std::shared_ptr<FinetunableRetriever> FinetunableRetriever::load_stream(
    std::istream& istream) {
  auto archive = ar::deserialize(istream);

  return std::shared_ptr<FinetunableRetriever>(
      new FinetunableRetriever(*archive));
}

}  // namespace thirdai::search