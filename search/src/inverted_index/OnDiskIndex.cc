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

inline bool isPruned(const rocksdb::Slice& value) {
  auto status = *reinterpret_cast<const TokenStatus*>(value.data());
  return status == TokenStatus::Pruned;
}

inline void appendTokenStatus(std::string& value, TokenStatus status) {
  value.append(reinterpret_cast<const char*>(&status), sizeof(TokenStatus));
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

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class AppendDocCounts : public rocksdb::AssociativeMergeOperator {
  /**
   * This merge operator appends a list of serialized DocCounts to another
   * existing list. Inputs are an existing_value and value are a serialized
   * array DocCounts preceded by a TokenStatus indicating if the token has been
   * pruned. The existing_value arg is the current value associated with the
   * key, whereas the value arg indicates the new DocCounts that are being
   * appended. The output is a new value in the same format as the inputs, only
   * with the DocCounts from each the input values.
   *
   * Since this operator needs to be associative, every value and addition is
   * preceded by a TokenStatus to ensure that the inputs and outputs of the
   * operator are in the same format, so that the order in which they are
   * executed does not matter.
   *
   * The TokenStatus exists to act as a tombstone in case the token is pruned.
   * Without the TokenStatus we would have to worry about a token being pruned,
   * then added back in a future doc, having the status allows us to distinguish
   * between tokens that were pruned, and tokens that were not yet seen by the
   * index.
   *
   *
   * Example:
   *
   * During indexing a batch of docs token T occurs in docs 0 and 1 with counts
   * of 10 and 11 respectively. Essentially this will result in a merge like
   * this:
   *   Merge(
   *      existing=[],
   *      update=[Status=Unpruned, (doc=1, cnt=10), (doc=2, cnt=11)]
   *   ) -> [Status=Unpruned, (doc=0, cnt=10), (doc=1, cnt=11)]
   *
   * Later, indexing more docs with token T occuring in doc 2 with count 12 will
   * result in a merge like this:
   *   Merge(
   *      existing=[Status=Unpruned, (doc=1, cnt=10), (doc=2, cnt=11)],
   *      update=[Status=Unpruned, (doc=2, cnt=12)]
   *   ) -> [Status=Unpruned, (doc=0, cnt=10), (doc=1, cnt=11), (doc=2, cnt=12)]
   *
   */
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

    if (isPruned(*existing_value) || isPruned(value)) {
      *new_value = std::string();
      new_value->reserve(sizeof(TokenStatus));
      appendTokenStatus(*new_value, TokenStatus::Pruned);
      return true;
    }

    // Note: this assumes that the doc_ids in the old and new values are
    // disjoint. This is true because we check for duplicate doc_ids during
    // indexing. If we add support for updates, then this needs to be modified
    // to merge the 2 values based on doc_id.

    // The +/- sizeof(TokenStatus) is because we are discarding the TokenStatus
    // from the second value because we only want 1 TokenStatus at the begining,
    // and we know that the value is the same in both TokenStatus's at this
    // point since neither is Pruned.
    *new_value = std::string();
    new_value->reserve(existing_value->size() + value.size() -
                       sizeof(TokenStatus));

    new_value->append(existing_value->data(), existing_value->size());
    new_value->append(value.data() + sizeof(TokenStatus),
                      value.size() - sizeof(TokenStatus));

    return true;
  }

  const char* Name() const override { return "AppendDocTokenCount"; }
};

class IncrementCounter : public rocksdb::AssociativeMergeOperator {
  /**
   * This merge operator is a simple counter operator, that will add the new
   * value to the existing value.
   */
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    int64_t counter = 0;
    if (existing_value) {
      if (!deserialize(*existing_value, counter)) {
        return false;
      }
    }

    int64_t increment = 0;
    if (!deserialize(value, increment)) {
      return false;
    }

    *new_value = std::string(sizeof(int64_t), 0);
    *reinterpret_cast<int64_t*>(new_value->data()) = counter + increment;

    return true;
  }

  const char* Name() const override { return "IncrementCounter"; }
};

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

  rocksdb::ColumnFamilyOptions counter_options;
  counter_options.merge_operator = std::make_shared<IncrementCounter>();

  rocksdb::ColumnFamilyOptions token_to_docs_options;
  token_to_docs_options.merge_operator = std::make_shared<AppendDocCounts>();

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName,
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("counters", counter_options),
      rocksdb::ColumnFamilyDescriptor("token_to_docs", token_to_docs_options),
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

void OnDiskIndex::storeDocLens(const std::vector<DocId>& ids,
                               const std::vector<uint32_t>& doc_lens) {
  int64_t sum_doc_lens = 0;

  rocksdb::Transaction* txn = startTransaction();

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const int64_t doc_len = doc_lens[i];

    const std::string key = docIdKey(doc_id);

    std::string unused_value;
    auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), _counters, key,
                                        &unused_value);
    if (get_status.ok()) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already indexed.");
    }
    if (!get_status.IsNotFound()) {
      raiseError(get_status, "txn get");
    }

    auto put_status =
        txn->Put(_counters, key,
                 rocksdb::Slice(reinterpret_cast<const char*>(&doc_len),
                                sizeof(doc_len)));
    if (!put_status.ok()) {
      raiseError(put_status, "txn put");
    }

    sum_doc_lens += doc_len;
  }

  auto status = txn->Commit();
  if (!status.ok()) {
    raiseError(status, "txn commit");
  }
  delete txn;

  updateNDocs(ids.size());
  updateSumDocLens(sum_doc_lens);
}

void OnDiskIndex::updateTokenToDocs(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_new_docs) {
  rocksdb::WriteBatch batch;

  for (const auto& [token, doc_counts] : token_to_new_docs) {
    rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

    const char* data_start = reinterpret_cast<const char*>(doc_counts.data());
    const char* data_end =
        reinterpret_cast<const char*>(doc_counts.data() + doc_counts.size());

    const size_t slice_len = data_end - data_start;
    if (slice_len != sizeof(DocCount) * doc_counts.size()) {
      throw std::invalid_argument("Alignment issue");
    }

    std::string value;
    value.reserve(sizeof(TokenStatus) + slice_len);
    appendTokenStatus(value, TokenStatus::Default);
    value.append(data_start, slice_len);

    // TODO(Nicholas): is it faster to just store the count per doc as a
    // unique key with <token>_<doc> -> cnt and then do prefix scans on
    // <token>_ to find the docs it occurs in?
    auto merge_status = batch.Merge(_token_to_docs, key, value);
    if (!merge_status.ok()) {
      raiseError(merge_status, "add merge to batch");
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError(status, "write batch");
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

void OnDiskIndex::incrementDocLens(
    const std::vector<DocId>& ids,
    const std::vector<uint32_t>& doc_len_increments) {
  rocksdb::WriteBatch batch;

  int64_t sum_new_lens = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    const int64_t doc_len = doc_len_increments[i];
    sum_new_lens += doc_len;

    auto merge_status =
        batch.Merge(_counters, docIdKey(ids[i]),
                    rocksdb::Slice(reinterpret_cast<const char*>(&doc_len),
                                   sizeof(doc_len)));
    if (!merge_status.ok()) {
      raiseError(merge_status, "add merge to batch");
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError(status, "write batch");
  }

  updateSumDocLens(sum_new_lens);
}

void OnDiskIndex::incrementDocTokenCounts(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_doc_updates) {
  rocksdb::Transaction* txn = startTransaction();

  for (const auto& [token, updates] : token_to_doc_updates) {
    rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

    std::string value;
    auto get_status =
        txn->GetForUpdate(rocksdb::ReadOptions(), _token_to_docs, key, &value);
    if (get_status.IsNotFound()) {
      appendTokenStatus(value, TokenStatus::Default);
    } else if (!get_status.ok()) {
      raiseError(get_status, "txn get");
    }

    for (const auto& update : updates) {
      auto* doc_counts = docCountPtr(value);
      const size_t docs_w_token = docsWithToken(value);

      const DocId doc_id = update.doc_id;
      auto* it =
          std::find_if(doc_counts, doc_counts + docs_w_token,
                       [doc_id](const auto& a) { return a.doc_id == doc_id; });
      if (it != doc_counts + docs_w_token) {
        it->count += update.count;
      } else {
        value.append(reinterpret_cast<const char*>(&update), sizeof(DocCount));
      }
    }

    auto put_status = txn->Put(_token_to_docs, key, value);
    if (!put_status.ok()) {
      raiseError(put_status, "txn put");
    }
  }

  auto status = txn->Commit();
  if (!status.ok()) {
    raiseError(status, "txn commit");
  }

  delete txn;
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
  auto status =
      _db->Merge(rocksdb::WriteOptions(), _counters, "n_docs",
                 rocksdb::Slice(reinterpret_cast<const char*>(&n_new_docs),
                                sizeof(int64_t)));
  if (!status.ok()) {
    raiseError(status, "merge");
  }
}

void OnDiskIndex::updateSumDocLens(int64_t sum_new_doc_lens) {
  auto status = _db->Merge(
      rocksdb::WriteOptions(), _counters, "sum_doc_lens",
      rocksdb::Slice(reinterpret_cast<const char*>(&sum_new_doc_lens),
                     sizeof(int64_t)));
  if (!status.ok()) {
    raiseError(status, "merge");
  }
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