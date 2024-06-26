#include "OnDiskIndex.h"
#include <dataset/src/utils/TokenEncoding.h>
#include <licensing/src/CheckLicense.h>
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::search {

struct __attribute__((packed)) DocCount {
  uint64_t doc_id;
  uint32_t count;
};

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class AppendDocTokenCount final : public rocksdb::AssociativeMergeOperator {
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const final {
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

  const char* Name() const final { return "AppendDocTokenCount"; }
};

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

  auto doc_lens_and_occurences = countTokenOccurences(docs);

  rocksdb::WriteBatch batch;

  uint64_t sum_doc_lens = 0;

  for (size_t i = 0; i < docs.size(); i++) {
    const DocId doc_id = ids[i];
    const uint32_t doc_len = doc_lens_and_occurences[i].first;
    const auto& occurences = doc_lens_and_occurences[i].second;

    if (containsDoc(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    for (const auto& [token, cnt] : occurences) {
      rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

      DocCount data{doc_id, cnt};

      // TODO(Nicholas): is it faster to just store the count per doc as a
      // unique key with <token>_<doc> -> cnt and then do prefix scans on
      // <token>_ to find the docs it occurs in?
      auto merge_status = batch.Merge(
          key,
          rocksdb::Slice(reinterpret_cast<const char*>(&data), sizeof(data)));
      if (!merge_status.ok()) {
        throw std::invalid_argument("merge failed");
      }
    }

    sum_doc_lens += doc_len;
    auto put_status =
        batch.Put(docIdKey(doc_id),
                  rocksdb::Slice(reinterpret_cast<const char*>(&doc_len),
                                 sizeof(doc_len)));
    if (!put_status.ok()) {
      throw std::invalid_argument("put failed");
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    throw std::invalid_argument("write failed");
  }

  updateNDocsAndAvgLen(sum_doc_lens, ids.size());
}

std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>>
OnDiskIndex::countTokenOccurences(const std::vector<std::string>& docs) const {
  std::vector<std::pair<uint32_t, std::unordered_map<uint32_t, uint32_t>>>
      token_counts(docs.size());

#pragma omp parallel for default(none) shared(docs, token_counts)
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_tokens = tokenize(docs[i]);

    std::unordered_map<uint32_t, uint32_t> counts;
    for (const auto& token : doc_tokens) {
      counts[token]++;
    }
    token_counts[i] = {doc_tokens.size(), std::move(counts)};
  }

  return token_counts;
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

  auto [n_docs, avg_doc_len] = getNDocsAndAvgLen();

  std::vector<std::pair<size_t, float>> token_indexes_and_idfs;
  token_indexes_and_idfs.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    if (statuses[i].ok()) {
      assert(values[i].size() % sizeof(DocCount) == 0);

      const size_t docs_w_token = values[i].size() / sizeof(DocCount);
      const float token_idf = idf(n_docs, docs_w_token);
      if (token_idf < _idf_cutoff) {
        token_indexes_and_idfs.emplace_back(i, token_idf);
      }
    } else if (!statuses[i].IsNotFound()) {
      throw std::invalid_argument("MultiGet failed");
    }
  }

  std::sort(token_indexes_and_idfs.begin(), token_indexes_and_idfs.end(),
            HighestScore<size_t>{});

  std::unordered_map<DocId, float> doc_scores;  // TODO(): cache doc lens here

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
    throw std::invalid_argument("Check failed");
  }

  return status.ok();
}

uint32_t OnDiskIndex::getDocLen(DocId doc_id) {
  std::string value;
  auto status = _db->Get(rocksdb::ReadOptions(), docIdKey(doc_id), &value);

  if (!status.ok()) {
    throw std::invalid_argument("Get failed");
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

std::vector<uint32_t> OnDiskIndex::tokenize(const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

}  // namespace thirdai::search