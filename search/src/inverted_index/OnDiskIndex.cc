#include "OnDiskIndex.h"
#include <bolt/src/utils/Timer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <licensing/src/CheckLicense.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/utilities/write_batch_with_index.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/MongoDbAdapter.h>
#include <search/src/inverted_index/RocksDbAdapter.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

OnDiskIndex::OnDiskIndex(const std::string& db_name, const DBAdapterConfig &db_adapter_config, const IndexConfig& config)
    : _max_docs_to_score(config.max_docs_to_score),
      _max_token_occurrence_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b),
      _tokenizer(config.tokenizer) {
  if (db_adapter_config.db_adapter == "rocksdb") {
    _db = std::make_unique<RocksDbAdapter>(db_name);
  } else if (db_adapter_config.db_adapter == "mongodb"){
    _db = std::make_unique<MongoDbAdapter>(db_adapter_config.db_uri, db_name,
                                           db_adapter_config.batch_size);
  } else{
        throw std::invalid_argument("Invalid 'db_adapter' value. The 'db_adapter' must be either 'rocksdb' or 'mongodb'. Please ensure you are using one of these supported database types.");
  }
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

template <typename T>
struct HighestScore {
  using Item = std::pair<T, float>;
  bool operator()(const Item& a, const Item& b) const {
    return a.second > b.second;
  }
};

std::vector<DocScore> OnDiskIndex::query(const std::string& query,
                                         uint32_t k) const {
  auto query_tokens = tokenize(query);

  auto doc_count_iterators = _db->lookupDocs(query_tokens);

  const uint64_t n_docs = _db->getNDocs();
  const float avg_doc_len = static_cast<float>(_db->getSumDocLens()) / n_docs;

  const uint64_t max_docs_with_token =
      std::max<uint64_t>(_max_token_occurrence_frac * n_docs, 1000);

  std::vector<std::pair<size_t, float>> token_indexes_and_idfs;
  token_indexes_and_idfs.reserve(doc_count_iterators.size());
  for (size_t i = 0; i < doc_count_iterators.size(); i++) {
    size_t docs_w_token = doc_count_iterators[i].len();

    if (docs_w_token < max_docs_with_token) {
      const float token_idf = idf(n_docs, docs_w_token);
      token_indexes_and_idfs.emplace_back(i, token_idf);
    }
  }

  std::sort(token_indexes_and_idfs.begin(), token_indexes_and_idfs.end(),
            HighestScore<size_t>{});

  // TODO(Nicholas): cache doc lens with score, to avoid duplicate lookups
  std::unordered_map<DocId, float> doc_scores;

  for (const auto& [token_index, token_idf] : token_indexes_and_idfs) {
    for (const auto& doc_count : doc_count_iterators[token_index]) {
      const DocId doc_id = doc_count.doc_id;

      if (doc_scores.size() < _max_docs_to_score || doc_scores.count(doc_id)) {
        const uint32_t doc_len = _db->getDocLen(doc_id);
        const float score =
            bm25(token_idf, doc_count.count, doc_len, avg_doc_len);
        doc_scores[doc_count.doc_id] += score;
      }
    }
  }

  return InvertedIndex::topk(doc_scores, k);
}

std::vector<HashedToken> OnDiskIndex::tokenize(const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

}  // namespace thirdai::search