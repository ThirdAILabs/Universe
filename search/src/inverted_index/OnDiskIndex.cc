#include "OnDiskIndex.h"
#include <bolt/src/utils/Timer.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <licensing/src/CheckLicense.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/write_batch_with_index.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/RocksDbAdapter.h>
#include <search/src/inverted_index/Utils.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

namespace {

std::string dbName(const std::string& path) {
  return std::filesystem::path(path) / "db";
}

std::string metadataPath(const std::string& path) {
  return std::filesystem::path(path) / "metadata";
}

}  // namespace

OnDiskIndex::OnDiskIndex(const std::string& save_path,
                         const IndexConfig& config)
    : OnDiskIndex(save_path, nullptr, config) {}

OnDiskIndex::OnDiskIndex(const std::string& save_path,
                         std::unique_ptr<DbAdapter> db,
                         const IndexConfig& config)
    : _db(std::move(db)),
      _save_path(save_path),
      _max_docs_to_score(config.max_docs_to_score),
      _max_token_occurrence_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b),
      _tokenizer(config.tokenizer) {
  licensing::checkLicense();

  createDirectory(save_path);

  if (!_db) {
    _db = std::make_unique<RocksDbAdapter>(dbName(_save_path));
  }

  auto metadata = ar::Map::make();
  metadata->set("config", config.toArchive());
  metadata->set("db_type", ar::str(_db->type()));

  auto metadata_file = dataset::SafeFileIO::ofstream(metadataPath(save_path));
  ar::serialize(metadata, metadata_file);
}

void OnDiskIndex::index(const std::vector<DocId>& ids,
                        const std::vector<std::string>& docs) {
  if (ids.size() != docs.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  licensing::entitlements().verifyNoDataSourceRetrictions();

  auto [doc_lens, token_counts] = countTokenOccurences(docs);

  _db->storeDocLens(ids, doc_lens);

  std::unordered_map<HashedToken, std::vector<DocCount>> coalesced_counts;
  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];

    for (const auto& [token, count] : token_counts[i]) {
      coalesced_counts[token].emplace_back(doc_id, count);
    }
  }

  _db->updateTokenToDocs(coalesced_counts);
}

std::pair<std::vector<uint32_t>,
          std::vector<std::unordered_map<HashedToken, uint32_t>>>
OnDiskIndex::countTokenOccurences(const std::vector<std::string>& docs) const {
  std::vector<uint32_t> doc_lens(docs.size());
  std::vector<std::unordered_map<uint32_t, uint32_t>> token_counts(docs.size());

#pragma omp parallel for default(none) shared(docs, doc_lens, token_counts)
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_tokens = tokenize(docs[i]);
    doc_lens[i] = doc_tokens.size();

    std::unordered_map<uint32_t, uint32_t> counts;
    for (const auto& token : doc_tokens) {
      counts[token]++;
    }
    token_counts[i] = std::move(counts);
  }

  return {std::move(doc_lens), std::move(token_counts)};
}

std::unordered_map<DocId, float> OnDiskIndex::scoreDocuments(
    const std::string& query) const {
  auto query_tokens = tokenize(query);

  auto doc_counts = _db->lookupDocs(query_tokens);

  const uint64_t n_docs = _db->getNDocs();
  const float avg_doc_len = static_cast<float>(_db->getSumDocLens()) / n_docs;

  const uint64_t max_docs_with_token =
      std::max<uint64_t>(_max_token_occurrence_frac * n_docs, 1000);

  std::vector<std::pair<size_t, float>> token_indexes_and_idfs;
  token_indexes_and_idfs.reserve(doc_counts.size());
  for (size_t i = 0; i < doc_counts.size(); i++) {
    const auto docs_w_token = doc_counts[i].size();
    if (docs_w_token < max_docs_with_token) {
      const float token_idf = idf(n_docs, docs_w_token);
      token_indexes_and_idfs.emplace_back(i, token_idf);
    }
  }

  std::sort(token_indexes_and_idfs.begin(), token_indexes_and_idfs.end(),
            HighestScore<size_t>{});

  std::unordered_map<DocId, float> doc_scores;

  // This is used to cache the lens for docs that have already been seen to
  // avoid the DB lookup. This speeds up query processing.
  std::unordered_map<DocId, uint32_t> doc_lens;

  for (const auto& [token_index, token_idf] : token_indexes_and_idfs) {
    const auto& counts = doc_counts[token_index];
    const size_t docs_w_token = counts.size();

    for (size_t i = 0; i < docs_w_token; i++) {
      const DocId doc_id = counts[i].doc_id;

      if (doc_scores.count(doc_id)) {
        const float score =
            bm25(token_idf, counts[i].count, doc_lens.at(doc_id), avg_doc_len);
        doc_scores[counts[i].doc_id] += score;
      } else if (doc_scores.size() < _max_docs_to_score) {
        uint32_t doc_len;
        if (doc_lens.count(doc_id)) {
          doc_len = doc_lens.at(doc_id);
        } else {
          doc_len = _db->getDocLen(doc_id);
          doc_lens[doc_id] = doc_len;
        }

        const float score =
            bm25(token_idf, counts[i].count, doc_len, avg_doc_len);
        doc_scores[counts[i].doc_id] += score;
      }
    }
  }

  return doc_scores;
}

std::vector<DocScore> OnDiskIndex::query(const std::string& query, uint32_t k,
                                         bool parallelize) const {
  (void)parallelize;

  auto doc_scores = scoreDocuments(query);

  return InvertedIndex::topk(doc_scores, k);
}

std::vector<DocScore> OnDiskIndex::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k, bool parallelize) const {
  (void)parallelize;

  auto doc_scores = scoreDocuments(query);

  const HighestScore<DocId> cmp;
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);

  for (const auto& [doc, score] : doc_scores) {
    if (candidates.count(doc) &&
        (top_scores.size() < k || top_scores.front().second < score)) {
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

void OnDiskIndex::remove(const std::vector<DocId>& ids) {
  _db->removeDocs({ids.begin(), ids.end()});
}

void OnDiskIndex::prune() {
  uint64_t n_docs = _db->getNDocs();

  const uint64_t max_docs_with_token =
      std::max<uint64_t>(_max_token_occurrence_frac * n_docs, 1000);

  _db->prune(max_docs_with_token);
}

std::vector<HashedToken> OnDiskIndex::tokenize(const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

void OnDiskIndex::save(const std::string& new_save_path) const {
  licensing::entitlements().verifySaveLoad();

  createDirectory(new_save_path);

  std::filesystem::copy(metadataPath(_save_path), new_save_path);

  _db->save(dbName(new_save_path));
}

std::shared_ptr<OnDiskIndex> OnDiskIndex::load(const std::string& save_path) {
  licensing::entitlements().verifySaveLoad();

  auto metadata_file = dataset::SafeFileIO::ifstream(metadataPath(save_path));
  auto metadata = ar::deserialize(metadata_file);

  auto config = IndexConfig::fromArchive(*metadata->get("config"));
  std::string db_type = metadata->str("db_type");

  std::unique_ptr<DbAdapter> db;
  if (db_type == "rocksdb") {
    db = std::make_unique<RocksDbAdapter>(dbName(save_path));
  } else {
    throw std::invalid_argument("Invalid db_type '" + db_type + "'.");
  }

  return std::make_shared<OnDiskIndex>(save_path, std::move(db), config);
}

}  // namespace thirdai::search