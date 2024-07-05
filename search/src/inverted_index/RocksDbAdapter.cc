#include "RocksDbAdapter.h"
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <filesystem>

namespace thirdai::search {

void raiseError(const std::string& op, const rocksdb::Status& status) {
  throw std::runtime_error(op + " failed with error: " + status.ToString() +
                           ".");
}

std::string docIdKey(uint64_t doc_id) {
  return "doc_" + std::to_string(doc_id);
}

// Using uint32_t since this will be prepended to doc counts, and so uint32_t
// ensures that it is still half-word aligned.
enum class TokenStatus : uint32_t {
  Default = 0,
  Pruned = 1,
};

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class AppendDocCounts : public rocksdb::AssociativeMergeOperator {
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    if (!existing_value) {
      *new_value = std::string(sizeof(TokenStatus) + value.size(), 0);

      *reinterpret_cast<TokenStatus*>(new_value->data()) = TokenStatus::Default;

      std::copy(value.data(), value.data() + value.size(),
                new_value->data() + sizeof(TokenStatus));

      return true;
    }

    auto status = *reinterpret_cast<const TokenStatus*>(existing_value->data());
    if (status == TokenStatus::Pruned) {
      return true;
    }

    // Note: this assumes that the doc_ids in the old and new values are
    // disjoint. This is true because we check for duplicate doc_ids during
    // indexing. If we add support for updates, then this needs to be modified
    // to merge the 2 values based on doc_id.

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

template <typename T>
bool deserialize(const rocksdb::Slice& value, T& output) {
  if (value.size() != sizeof(T)) {
    return false;
  }
  output = *reinterpret_cast<const T*>(value.data());
  return true;
}

class IncrementCounter : public rocksdb::AssociativeMergeOperator {
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    uint64_t counter = 0;
    if (existing_value) {
      if (!deserialize(*existing_value, counter)) {
        return false;
      }
    }

    uint64_t increment = 0;
    if (!deserialize(value, increment)) {
      return false;
    }

    *new_value = std::string(sizeof(uint64_t), 0);
    *reinterpret_cast<uint64_t*>(new_value->data()) = counter + increment;

    return true;
  }

  const char* Name() const override { return "IncrementCounter"; }
};

RocksDbAdapter::RocksDbAdapter(const std::string& save_path)
    : _save_path(save_path) {
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

  auto status =
      rocksdb::TransactionDB::Open(options, rocksdb::TransactionDBOptions(),
                                   save_path, column_families, &handles, &_db);
  if (!status.ok()) {
    raiseError("Database creation", status);
  }

  if (handles.size() != 3) {
    throw std::runtime_error("Expected 2 handles to be created. Received " +
                             std::to_string(handles.size()) + " handles.");
  }

  _default = handles[0];
  _counters = handles[1];
  _token_to_docs = handles[2];

  updateNDocs(0);
  updateSumDocLens(0);
}

void RocksDbAdapter::storeDocLens(const std::vector<DocId>& ids,
                                  const std::vector<uint32_t>& doc_lens,
                                  bool check_for_existing) {
  uint64_t sum_doc_lens = 0;

  rocksdb::Transaction* txn = _db->BeginTransaction(rocksdb::WriteOptions());

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const uint32_t doc_len = doc_lens[i];

    const std::string key = docIdKey(doc_id);

    // TODO(Nicholas): decrement count delta
    if (check_for_existing) {
      std::string unused_value;
      auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), _counters,
                                          key, &unused_value);
      if (get_status.ok()) {
        throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                                 " is already indexed.");
      }
      if (!get_status.IsNotFound()) {
        raiseError("check for doc", get_status);
      }
    }

    auto put_status =
        txn->Put(_counters, key,
                 rocksdb::Slice(reinterpret_cast<const char*>(&doc_len),
                                sizeof(doc_len)));
    if (!put_status.ok()) {
      raiseError("Add write to batch", put_status);
    }

    sum_doc_lens += doc_len;
  }

  auto status = txn->Commit();
  if (!status.ok()) {
    raiseError("Write txn commit", status);
  }

  updateNDocs(ids.size());
  updateSumDocLens(sum_doc_lens);
}

void RocksDbAdapter::updateTokenToDocs(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_new_docs) {
  rocksdb::WriteBatch batch;

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
    auto merge_status = batch.Merge(_token_to_docs, key, value);
    if (!merge_status.ok()) {
      raiseError("Add merge to batch", merge_status);
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError("Write batch commit", status);
  }
}

inline bool isPruned(const std::string& value) {
  auto status = *reinterpret_cast<const TokenStatus*>(value.data());
  return status == TokenStatus::Pruned;
}

inline size_t docsWithToken(const rocksdb::Slice& value) {
  assert((value.size() - sizeof(TokenStatus)) % sizeof(DocCount) == 0);
  return (value.size() - sizeof(TokenStatus)) / sizeof(DocCount);
}

inline const DocCount* docCountPtr(const std::string& value) {
  return reinterpret_cast<const DocCount*>(value.data() + sizeof(TokenStatus));
}

std::vector<std::vector<DocCount>> RocksDbAdapter::lookupDocs(
    const std::vector<HashedToken>& query_tokens) const {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());
  std::vector<std::string> values;

  for (const auto& token : query_tokens) {
    keys.emplace_back(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  values.resize(keys.size());
  std::vector<rocksdb::ColumnFamilyHandle*> handles(keys.size(),
                                                    _token_to_docs);
  auto statuses = _db->MultiGet(rocksdb::ReadOptions(), handles, keys, &values);

  std::vector<std::vector<DocCount>> results;
  results.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    if (statuses[i].ok()) {
      if (!isPruned(values[i])) {
        const DocCount* ptr = docCountPtr(values[i]);
        const size_t n_docs_w_token = docsWithToken(values[i]);
        results.emplace_back(ptr, ptr + n_docs_w_token);
      }
    } else if (!statuses[i].IsNotFound()) {
      raiseError("DB batch get", statuses[i]);
    }
  }

  return results;
}

void RocksDbAdapter::prune(uint64_t max_docs_with_token) {
  rocksdb::Iterator* iter =
      _db->NewIterator(rocksdb::ReadOptions(), _token_to_docs);

  rocksdb::WriteBatch batch;
  TokenStatus prune = TokenStatus::Pruned;

  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    if (docsWithToken(iter->value()) > max_docs_with_token) {
      auto write_status = batch.Put(
          _token_to_docs, iter->key(),
          rocksdb::Slice(reinterpret_cast<const char*>(&prune), sizeof(prune)));
      if (!write_status.ok()) {
        raiseError("add to write batch", write_status);
      }
    }
  }

  delete iter;

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError("prune", status);
  }

  auto compact_status = _db->CompactRange(rocksdb::CompactRangeOptions(),
                                          _token_to_docs, nullptr, nullptr);
  if (!compact_status.ok()) {
    raiseError("compact", compact_status);
  }
}

uint32_t RocksDbAdapter::getDocLen(DocId doc_id) const {
  std::string value;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, docIdKey(doc_id), &value);
  if (!status.ok()) {
    raiseError("DB read", status);
  }

  uint32_t len;
  if (!deserialize(value, len)) {
    throw std::invalid_argument("document length is corrupted");
  }

  return len;
}

uint64_t RocksDbAdapter::getNDocs() const {
  std::string serialized;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, "n_docs", &serialized);
  if (!status.ok()) {
    raiseError("DB read", status);
  }

  uint64_t ndocs;
  if (!deserialize(serialized, ndocs)) {
    throw std::invalid_argument("Value of n_docs is corrupted.");
  }
  return ndocs;
}

uint64_t RocksDbAdapter::getSumDocLens() const {
  std::string serialized;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, "sum_doc_lens", &serialized);
  if (!status.ok()) {
    raiseError("DB read", status);
  }

  uint64_t sum_doc_lens;
  if (!deserialize(serialized, sum_doc_lens)) {
    throw std::invalid_argument("Value of sum_doc_lens is corrupted.");
  }
  return sum_doc_lens;
}

void RocksDbAdapter::updateNDocs(uint64_t n_new_docs) {
  auto status =
      _db->Merge(rocksdb::WriteOptions(), _counters, "n_docs",
                 rocksdb::Slice(reinterpret_cast<const char*>(&n_new_docs),
                                sizeof(uint64_t)));
  if (!status.ok()) {
    raiseError("DB merge", status);
  }
}

void RocksDbAdapter::updateSumDocLens(uint64_t sum_new_doc_lens) {
  auto status = _db->Merge(
      rocksdb::WriteOptions(), _counters, "sum_doc_lens",
      rocksdb::Slice(reinterpret_cast<const char*>(&sum_new_doc_lens),
                     sizeof(uint64_t)));
  if (!status.ok()) {
    raiseError("DB merge", status);
  }
}

void RocksDbAdapter::save(const std::string& save_path) const {
  std::filesystem::copy(_save_path, save_path,
                        std::filesystem::copy_options::recursive);
}

RocksDbAdapter::~RocksDbAdapter() {
  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_counters);
  _db->DestroyColumnFamilyHandle(_token_to_docs);
  _db->Close();
  delete _db;
}

}  // namespace thirdai::search