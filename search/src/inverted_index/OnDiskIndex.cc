#include "OnDiskIndex.h"
#include <bolt/src/utils/Timer.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <licensing/src/CheckLicense.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

void raiseError(const std::string& op, const rocksdb::Status& status) {
  throw std::runtime_error(op + " failed with error: " + status.ToString() +
                           ".");
}

// The following might be more memory efficient.
// struct __attribute__((packed)) DocCount {
struct DocCount {
  DocCount(DocId doc_id, uint32_t count) : doc_id(doc_id), count(count) {}

  DocId doc_id;
  uint32_t count;
};

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class AppendDocTokenCount : public rocksdb::AssociativeMergeOperator {
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    if (!existing_value) {
      *new_value = value.ToString();
      return true;
    }

    // TODO(Nicholas): check for doc already existing?

    *new_value = std::string(existing_value->size() + value.size(), 0);

    std::copy(existing_value->data(),
              existing_value->data() + existing_value->size(),
              new_value->data());
    std::copy(value.data(), value.data() + value.size(),
              new_value->data() + existing_value->size());

    return true;
  }

  const char* Name() const override { return "AppendDocTokenCount"; }
};

OnDiskIndex::OnDiskIndex(const std::string& db_name) {
  rocksdb::Options options;
  options.create_if_missing = true;

  // options.table_factory.reset(rocksdb::NewCuckooTableFactory());
  options.merge_operator = std::make_shared<AppendDocTokenCount>();

  rocksdb::Status status = rocksdb::DB::Open(options, db_name, &_db);
  if (!status.ok()) {
    raiseError("Database creation", status);
  }
}

std::string docIdKey(uint64_t doc_id) {
  return "doc_" + std::to_string(doc_id);
}

void OnDiskIndex::index(const std::vector<DocId>& ids,
                        const std::vector<std::string>& docs) {
  if (ids.size() != docs.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  licensing::entitlements().verifyNoDataSourceRetrictions();

  auto [doc_lens, token_counts] = countTokenOccurences(docs);

  storeDocLens(ids, doc_lens);

  updateTokenToDocs(ids, token_counts);

  // rocksdb::WriteBatch batch;
  // for (size_t i = 0; i < docs.size(); i++) {
  //   const DocId doc_id = ids[i];
  //   const auto& occurences = token_counts[i];

  //   for (const auto& [token, cnt] : occurences) {
  //     rocksdb::Slice key(reinterpret_cast<const char*>(&token),
  //     sizeof(token));

  //     DocCount data{doc_id, cnt};

  //     // TODO(Nicholas): is it faster to just store the count per doc as a
  //     // unique key with <token>_<doc> -> cnt and then do prefix scans on
  //     // <token>_ to find the docs it occurs in?
  //     auto merge_status = batch.Merge(
  //         key,
  //         rocksdb::Slice(reinterpret_cast<const char*>(&data),
  //         sizeof(data)));
  //     if (!merge_status.ok()) {
  //       raiseError("Add merge to batch", merge_status);
  //     }
  //   }
  // }

  // auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  // if (!status.ok()) {
  //   raiseError("Write batch commit", status);
  // }
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

void OnDiskIndex::storeDocLens(const std::vector<DocId>& ids,
                               const std::vector<uint32_t>& doc_lens) {
  uint64_t sum_doc_lens = 0;
  rocksdb::WriteBatch batch;

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const uint32_t doc_len = doc_lens[i];

    if (containsDoc(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    auto put_status =
        batch.Put(docIdKey(doc_id),
                  rocksdb::Slice(reinterpret_cast<const char*>(&doc_len),
                                 sizeof(doc_len)));
    if (!put_status.ok()) {
      raiseError("Add write to batch", put_status);
    }

    sum_doc_lens += doc_len;
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError("Write batch commit", status);
  }

  updateNDocsAndAvgLen(/*sum_new_doc_lens=*/sum_doc_lens,
                       /*n_new_docs=*/ids.size());
}

void OnDiskIndex::updateTokenToDocs(
    const std::vector<DocId>& ids,
    const std::vector<std::unordered_map<HashedToken, uint32_t>>&
        token_counts) {
  std::unordered_map<HashedToken, std::vector<DocCount>> coalesced_counts;
  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];

    for (const auto& [token, count] : token_counts[i]) {
      coalesced_counts[token].emplace_back(doc_id, count);
    }
  }

  rocksdb::WriteBatch batch;

  for (const auto& [token, doc_counts] : coalesced_counts) {
    rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

    const char* data_start = reinterpret_cast<const char*>(doc_counts.data());
    const char* data_end =
        reinterpret_cast<const char*>(doc_counts.data() + doc_counts.size());

    size_t slice_len = data_end - data_start;
    if (slice_len != sizeof(DocCount) * doc_counts.size()) {
      throw std::invalid_argument("Alignment issue");
    }

    rocksdb::Slice value(data_start, slice_len);

    // TODO(Nicholas): is it faster to just store the count per doc as a
    // unique key with <token>_<doc> -> cnt and then do prefix scans on
    // <token>_ to find the docs it occurs in?
    auto merge_status = batch.Merge(key, value);
    if (!merge_status.ok()) {
      raiseError("Add merge to batch", merge_status);
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError("Write batch commit", status);
  }
}

template <typename T>
struct HighestScore {
  using Item = std::pair<T, float>;
  bool operator()(const Item& a, const Item& b) const {
    return a.second > b.second;
  }
};

std::vector<DocScore> OnDiskIndex::query(const std::string& query, uint32_t k) {
  auto query_tokens = tokenize(query);

  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());
  std::vector<std::string> values;

  for (const auto& token : query_tokens) {
    keys.emplace_back(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  values.resize(keys.size());

  auto statuses = _db->MultiGet(rocksdb::ReadOptions(), keys, &values);

  const auto [n_docs, avg_doc_len] = getNDocsAndAvgLen();
  const uint64_t max_docs_with_token =
      std::max<uint64_t>(_idf_cutoff_frac * n_docs, 1000);

  std::vector<std::pair<size_t, float>> token_indexes_and_idfs;
  token_indexes_and_idfs.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    if (statuses[i].ok()) {
      assert(values[i].size() % sizeof(DocCount) == 0);

      const size_t docs_w_token = values[i].size() / sizeof(DocCount);

      if (docs_w_token < max_docs_with_token) {
        const float token_idf = idf(n_docs, docs_w_token);
        token_indexes_and_idfs.emplace_back(i, token_idf);
      }
    } else if (!statuses[i].IsNotFound()) {
      raiseError("DB batch get", statuses[i]);
    }
  }

  std::sort(token_indexes_and_idfs.begin(), token_indexes_and_idfs.end(),
            HighestScore<size_t>{});

  // TODO(Nicholas): cache doc lens with score, to avoid duplicate lookups
  std::unordered_map<DocId, float> doc_scores;

  for (const auto& [token_index, token_idf] : token_indexes_and_idfs) {
    const DocCount* counts =
        reinterpret_cast<const DocCount*>(values[token_index].data());
    const size_t docs_w_token = values[token_index].size() / sizeof(DocCount);

    for (size_t i = 0; i < docs_w_token; i++) {
      const DocId doc_id = counts[i].doc_id;

      if (doc_scores.size() < _max_docs_to_score || doc_scores.count(doc_id)) {
        const uint32_t doc_len = getDocLen(doc_id);
        const float score =
            bm25(token_idf, counts[i].count, doc_len, avg_doc_len);
        doc_scores[counts[i].doc_id] += score;
      }
    }
  }

  return InvertedIndex::topk(doc_scores, k);
}

bool OnDiskIndex::containsDoc(DocId doc_id) const {
  std::string value;
  auto status = _db->Get(rocksdb::ReadOptions(), docIdKey(doc_id), &value);

  if (!status.ok() && !status.IsNotFound()) {
    raiseError("DB read", status);
  }

  return status.ok();
}

OnDiskIndex::~OnDiskIndex() {
  _db->Close();
  delete _db;
}

uint32_t OnDiskIndex::getDocLen(DocId doc_id) {
  std::string value;
  auto status = _db->Get(rocksdb::ReadOptions(), docIdKey(doc_id), &value);
  if (!status.ok()) {
    raiseError("DB read", status);
  }

  assert(value.size() == sizeof(uint32_t));

  return *reinterpret_cast<const uint32_t*>(value.data());
}

std::pair<uint64_t, float> OnDiskIndex::getNDocsAndAvgLen() {
  std::shared_lock<std::shared_mutex> lock(_mutex);
  return {_n_docs, _avg_doc_len};
}

void OnDiskIndex::updateNDocsAndAvgLen(uint64_t sum_new_doc_lens,
                                       uint64_t n_new_docs) {
  std::unique_lock<std::shared_mutex> lock(_mutex);

  _n_docs += n_new_docs;
  _sum_doc_lens += sum_new_doc_lens;
  _avg_doc_len = static_cast<float>(_sum_doc_lens) / _n_docs;
}

std::vector<HashedToken> OnDiskIndex::tokenize(const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

}  // namespace thirdai::search