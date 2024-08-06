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
#include <search/src/inverted_index/BM25.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Utils.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::search {

namespace {

std::string dbName(const std::string& path) {
  return (std::filesystem::path(path) / "db").string();
}

std::string metadataPath(const std::string& path) {
  return (std::filesystem::path(path) / "metadata").string();
}

inline void raiseError(const rocksdb::Status& status, const std::string& op) {
  throw std::runtime_error(status.ToString() + op);
}

// Using uint32_t since this will be prepended to doc counts, and so uint32_t
// ensures that it is still half-word aligned.
enum class TokenStatus : uint32_t {
  Default = 0,
  Pruned = 1,
};

inline std::string docIdKey(DocId doc_id) {
  // We are prependeding the letter D to the keys so that we could do a prefix
  // scan to iterate over the different doc_ids in the database if needed. We
  // are serializing the numbers instead of converting to a string because it
  // will be more space efficient at larger scale.
  std::string key;
  key.reserve(sizeof(DocId) + 1);
  key.append("D");
  key.append(reinterpret_cast<const char*>(&doc_id), sizeof(DocId));
  return key;
}

template <typename T>
inline bool deserialize(const rocksdb::Slice& value, T& output) {
  if (value.size() != sizeof(T)) {
    return false;
  }
  output = *reinterpret_cast<const T*>(value.data());
  return true;
}

inline int64_t deserialize(const rocksdb::Slice& value) {
  if (value.size() != sizeof(int64_t)) {
    throw std::runtime_error("Corruption in serialized value.");
  }
  return *reinterpret_cast<const int64_t*>(value.data());
}

inline bool isPruned(const rocksdb::Slice& value) {
  auto status = *reinterpret_cast<const TokenStatus*>(value.data());
  return status == TokenStatus::Pruned;
}

inline size_t docsWithToken(const rocksdb::Slice& value) {
  assert((value.size() - sizeof(TokenStatus)) % sizeof(DocCount) == 0);
  return (value.size() - sizeof(TokenStatus)) / sizeof(DocCount);
}

inline const DocCount* docCountPtr(const rocksdb::Slice& value) {
  return reinterpret_cast<const DocCount*>(value.data() + sizeof(TokenStatus));
}

inline DocCount* docCountPtr(std::string& value) {
  return reinterpret_cast<DocCount*>(value.data() + sizeof(TokenStatus));
}

}  // namespace

OnDiskIndex::OnDiskIndex(const std::string& save_path,
                         const IndexConfig& config, bool read_only)
    : _save_path(save_path),
      _max_docs_to_score(config.max_docs_to_score),
      _max_token_occurrence_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b),
      _tokenizer(config.tokenizer) {
  licensing::checkLicense();

  createDirectory(save_path);
  auto metadata = ar::Map::make();
  metadata->set("config", config.toArchive());

  auto metadata_file = dataset::SafeFileIO::ofstream(metadataPath(save_path));
  ar::serialize(metadata, metadata_file);

  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName,
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("counters",
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("token_to_docs",
                                      rocksdb::ColumnFamilyOptions()),
  };

  std::vector<rocksdb::ColumnFamilyHandle*> handles;

  rocksdb::Status open_status;
  if (!read_only) {
    open_status = rocksdb::TransactionDB::Open(
        options, rocksdb::TransactionDBOptions(), save_path, column_families,
        &handles, &_transaction_db);
    _db = _transaction_db;
  } else {
    open_status = rocksdb::DB::OpenForReadOnly(options, save_path,
                                               column_families, &handles, &_db);
    _transaction_db = nullptr;
  }

  if (!open_status.ok()) {
    raiseError(open_status, "open database");
  }

  if (handles.size() != 3) {
    throw std::runtime_error("Expected 3 handles to be created. Received " +
                             std::to_string(handles.size()) + " handles.");
  }

  _default = handles[0];
  _counters = handles[1];
  _token_to_docs = handles[2];

  if (!read_only) {
    updateNDocs(0);
    updateSumDocLens(0);
  }
}

std::unordered_map<HashedToken, std::vector<DocCount>> coalesceCounts(
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

  return coalesced_counts;
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

  auto coalesced_counts = coalesceCounts(ids, token_counts);

  updateTokenToDocs(coalesced_counts);
}

std::string storeDocLenAndCheckForExisting(const std::string& value,
                                           int64_t doc_len) {
  if (!value.empty()) {
    throw std::runtime_error("Document with id " + value +
                             " is already indexed.");
  }

  std::string new_value;
  new_value.reserve(sizeof(int64_t));
  new_value.append(reinterpret_cast<const char*>(&doc_len), sizeof(int64_t));
  return new_value;
}

void OnDiskIndex::storeDocLens(const std::vector<DocId>& ids,
                               const std::vector<uint32_t>& doc_lens) {
  int64_t sum_doc_lens = 0;

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const int64_t doc_len = doc_lens[i];

    const std::string key = docIdKey(doc_id);

    updateKey(_counters, key, doc_len, storeDocLenAndCheckForExisting);

    sum_doc_lens += doc_len;
  }

  updateNDocs(ids.size());
  updateSumDocLens(sum_doc_lens);
}

std::string concatTokenCounts(const std::string& value,
                              const rocksdb::Slice& delta) {
  if (value.empty()) {
    std::string new_value;
    new_value.reserve(sizeof(TokenStatus) + delta.size());

    TokenStatus status = TokenStatus::Default;
    new_value.append(reinterpret_cast<const char*>(&status),
                     sizeof(TokenStatus));

    new_value.append(delta.data(), delta.size());
    return new_value;
  }

  std::string new_value;
  new_value.reserve(value.size() + delta.size());
  new_value.append(value.data(), value.size());
  new_value.append(delta.data(), delta.size());

  return new_value;
}

void OnDiskIndex::updateTokenToDocs(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_new_docs) {
  for (const auto& [token, doc_counts] : token_to_new_docs) {
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
    updateKey(_token_to_docs, key, value, concatTokenCounts);
  }
}

void OnDiskIndex::update(const std::vector<DocId>& ids,
                         const std::vector<std::string>& extra_tokens) {
  if (ids.size() != extra_tokens.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  licensing::entitlements().verifyNoDataSourceRetrictions();

  auto [doc_lens, token_counts] = countTokenOccurences(extra_tokens);

  incrementDocLens(ids, doc_lens);

  auto coalesced_counts = coalesceCounts(ids, token_counts);

  incrementDocTokenCounts(coalesced_counts);
}

std::string incrementCounter(const std::string& value, const int64_t& delta) {
  int64_t counter = value.empty() ? 0 : deserialize(value);
  counter += delta;

  std::string new_value;
  new_value.reserve(sizeof(int64_t));
  new_value.append(reinterpret_cast<const char*>(&counter), sizeof(int64_t));
  return new_value;
}

void OnDiskIndex::incrementDocLens(
    const std::vector<DocId>& ids,
    const std::vector<uint32_t>& doc_len_increments) {
  int64_t sum_new_lens = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    const int64_t doc_len = doc_len_increments[i];
    sum_new_lens += doc_len;

    updateKey(_counters, docIdKey(ids[i]), doc_len, incrementCounter);
  }

  updateSumDocLens(sum_new_lens);
}

std::string incrementTokenCounts(const std::string& value,
                                 const rocksdb::Slice& delta) {
  if (value.empty()) {
    std::string new_value;
    new_value.reserve(sizeof(TokenStatus) + delta.size());
    *reinterpret_cast<TokenStatus*>(new_value.data()) = TokenStatus::Default;
    new_value.append(delta.data(), delta.size());
    return new_value;
  }

  std::string new_value;
  new_value.reserve(value.size() + delta.size());
  new_value.append(value.data(), value.size());
  new_value.append(delta.data(), delta.size());

  return new_value;
}

std::string mergeTokenCounts(const std::string& value,
                             const std::vector<DocCount>& updates) {
  std::string new_value = value;
  if (new_value.empty()) {
    TokenStatus status = TokenStatus::Default;
    new_value.append(reinterpret_cast<const char*>(&status),
                     sizeof(TokenStatus));
  }

  for (const auto& update : updates) {
    auto* doc_counts = docCountPtr(new_value);
    const size_t docs_w_token = docsWithToken(new_value);

    const DocId doc_id = update.doc_id;
    auto* it =
        std::find_if(doc_counts, doc_counts + docs_w_token,
                     [doc_id](const auto& a) { return a.doc_id == doc_id; });

    if (it != doc_counts + docs_w_token) {
      it->count += update.count;
    } else {
      new_value.append(reinterpret_cast<const char*>(&update),
                       sizeof(DocCount));
    }
  }
  return new_value;
}

void OnDiskIndex::incrementDocTokenCounts(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_doc_updates) {
  for (const auto& [token, updates] : token_to_doc_updates) {
    rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

    updateKey(_token_to_docs, key, updates, mergeTokenCounts);
  }
}

// TODO(Nicholas): This logic is similar to that of the regular inverted index,
// except it uses the hashes of the tokens in place of the tokens themselves.
// The regular index should be updated to do this as well, and then this logic
// can be consolidated into a helper function.
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

  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());

  for (const auto& token : query_tokens) {
    keys.emplace_back(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), _token_to_docs, keys.size(),
                keys.data(), values.data(), statuses.data());

  const int64_t n_docs = getNDocs();
  const float avg_doc_len = static_cast<float>(getSumDocLens()) / n_docs;

  const int64_t max_docs_with_token =
      std::max<int64_t>(_max_token_occurrence_frac * n_docs, 1000);

  std::vector<std::pair<size_t, float>> token_indexes_and_idfs;
  token_indexes_and_idfs.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    if (statuses[i].ok()) {
      if (!isPruned(values[i])) {
        const int64_t n_docs_w_token = docsWithToken(values[i]);
        if (n_docs_w_token < max_docs_with_token) {
          token_indexes_and_idfs.emplace_back(i, idf(n_docs, n_docs_w_token));
        }
      }
    } else if (!statuses[i].IsNotFound()) {
      raiseError(statuses[i], "batch get");
    }
  }

  std::sort(token_indexes_and_idfs.begin(), token_indexes_and_idfs.end(),
            HighestScore<size_t>{});

  std::unordered_map<DocId, float> doc_scores;

  // This is used to cache the lens for docs that have already been seen to
  // avoid the DB lookup. This speeds up query processing.
  std::unordered_map<DocId, uint32_t> doc_lens;

  const uint64_t query_len = token_indexes_and_idfs.size();

  for (const auto& [token_index, token_idf] : token_indexes_and_idfs) {
    const DocCount* counts = docCountPtr(values[token_index]);
    const size_t docs_w_token = docsWithToken(values[token_index]);

    for (size_t i = 0; i < docs_w_token; i++) {
      const DocId doc_id = counts[i].doc_id;

      if (doc_scores.count(doc_id)) {
        const float score =
            bm25(/*idf=*/token_idf, /*cnt_in_doc=*/counts[i].count,
                 /*doc_len=*/doc_lens.at(doc_id), /*avg_doc_len=*/avg_doc_len,
                 /*query_len=*/query_len, /*k1=*/_k1, /*b=*/_b);
        doc_scores[counts[i].doc_id] += score;
      } else if (doc_scores.size() < _max_docs_to_score) {
        uint32_t doc_len;
        if (doc_lens.count(doc_id)) {
          doc_len = doc_lens.at(doc_id);
        } else {
          doc_len = getDocLen(doc_id);
          doc_lens[doc_id] = doc_len;
        }

        const float score =
            bm25(/*idf=*/token_idf, /*cnt_in_doc=*/counts[i].count,
                 /*doc_len=*/doc_len, /*avg_doc_len=*/avg_doc_len,
                 /*query_len=*/query_len, /*k1=*/_k1, /*b=*/_b);
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

void OnDiskIndex::remove(const std::vector<DocId>& id_list) {
  std::unordered_set<DocId> ids(id_list.begin(), id_list.end());

  rocksdb::Transaction* txn = startTransaction();

  int64_t sum_deleted_len = 0;
  for (const auto& doc : ids) {
    std::string serialized_doc_len;
    auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), _counters,
                                        docIdKey(doc), &serialized_doc_len);
    int64_t doc_len;
    if (!deserialize(serialized_doc_len, doc_len)) {
      throw std::runtime_error("document length corrupted");
    }
    sum_deleted_len += doc_len;

    auto delete_status = txn->Delete(_counters, docIdKey(doc));
    if (!delete_status.ok()) {
      raiseError(delete_status, "txn delete");
    }
  }

  updateSumDocLens(-sum_deleted_len);
  updateNDocs(-ids.size());

  rocksdb::Iterator* iter =
      txn->GetIterator(rocksdb::ReadOptions(), _token_to_docs);

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    auto curr_value = iter->value();
    std::string new_value;
    new_value.reserve(curr_value.size());

    new_value.append(curr_value.data(), sizeof(TokenStatus));

    const DocCount* doc_counts = docCountPtr(curr_value);
    const size_t docs_w_token = docsWithToken(curr_value);

    for (size_t i = 0; i < docs_w_token; i++) {
      if (!ids.count(doc_counts[i].doc_id)) {
        new_value.append(reinterpret_cast<const char*>(doc_counts + i),
                         sizeof(DocCount));
      }
    }

    if (new_value.size() < curr_value.size()) {
      auto put_status = txn->Put(_token_to_docs, iter->key(), new_value);
      if (!put_status.ok()) {
        raiseError(put_status, "txn put");
      }
    }
  }

  delete iter;

  auto status = txn->Commit();
  if (!status.ok()) {
    raiseError(status, "txn commit");
  }

  delete txn;
}

void OnDiskIndex::prune() {
  int64_t n_docs = getNDocs();

  const int64_t max_docs_with_token =
      std::max<int64_t>(_max_token_occurrence_frac * n_docs, 1000);

  rocksdb::Iterator* iter =
      _db->NewIterator(rocksdb::ReadOptions(), _token_to_docs);

  rocksdb::WriteBatch batch;
  TokenStatus prune = TokenStatus::Pruned;

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    const int64_t docs_w_token = docsWithToken(iter->value());
    if (docs_w_token > max_docs_with_token) {
      auto write_status = batch.Put(
          _token_to_docs, iter->key(),
          rocksdb::Slice(reinterpret_cast<const char*>(&prune), sizeof(prune)));
      if (!write_status.ok()) {
        raiseError(write_status, "add put to batch");
      }
    }
  }

  delete iter;

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError(status, "write batch");
  }

  auto compact_status = _db->CompactRange(rocksdb::CompactRangeOptions(),
                                          _token_to_docs, nullptr, nullptr);
  if (!compact_status.ok()) {
    raiseError(compact_status, "compact");
  }
}

int64_t OnDiskIndex::getDocLen(DocId doc_id) const {
  std::string value;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, docIdKey(doc_id), &value);
  if (!status.ok()) {
    raiseError(status, "get");
  }

  int64_t len;
  if (!deserialize(value, len)) {
    throw std::invalid_argument("document length is corrupted");
  }

  return len;
}

int64_t OnDiskIndex::getNDocs() const {
  std::string serialized;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, "n_docs", &serialized);
  if (!status.ok()) {
    raiseError(status, "get");
  }

  int64_t ndocs;
  if (!deserialize(serialized, ndocs)) {
    throw std::invalid_argument("Value of n_docs is corrupted.");
  }
  return ndocs;
}

int64_t OnDiskIndex::getSumDocLens() const {
  std::string serialized;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, "sum_doc_lens", &serialized);
  if (!status.ok()) {
    raiseError(status, "get");
  }

  int64_t sum_doc_lens;
  if (!deserialize(serialized, sum_doc_lens)) {
    throw std::invalid_argument("Value of sum_doc_lens is corrupted.");
  }
  return sum_doc_lens;
}

void OnDiskIndex::updateNDocs(int64_t n_new_docs) {
  updateKey(_counters, "n_docs", n_new_docs, incrementCounter);
}

void OnDiskIndex::updateSumDocLens(int64_t sum_new_doc_lens) {
  updateKey(_counters, "sum_doc_lens", sum_new_doc_lens, incrementCounter);
}

std::vector<HashedToken> OnDiskIndex::tokenize(const std::string& text) const {
  auto tokens = _tokenizer->tokenize(text);
  return dataset::token_encoding::hashTokens(tokens);
}

void OnDiskIndex::save(const std::string& new_save_path) const {
  licensing::entitlements().verifySaveLoad();

  createDirectory(new_save_path);

  std::filesystem::copy(_save_path, new_save_path,
                        std::filesystem::copy_options::recursive);
}

std::shared_ptr<OnDiskIndex> OnDiskIndex::load(const std::string& save_path,
                                               bool read_only) {
  licensing::entitlements().verifySaveLoad();

  auto metadata_file = dataset::SafeFileIO::ifstream(metadataPath(save_path));
  auto metadata = ar::deserialize(metadata_file);

  auto config = IndexConfig::fromArchive(*metadata->get("config"));

  return std::make_shared<OnDiskIndex>(save_path, config, read_only);
}

OnDiskIndex::~OnDiskIndex() {
  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_counters);
  _db->DestroyColumnFamilyHandle(_token_to_docs);
  _db->Close();
  delete _db;
}

}  // namespace thirdai::search