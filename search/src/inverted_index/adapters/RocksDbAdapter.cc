#include "RocksDbAdapter.h"
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/adapters/RocksDbUtils.h>
#include <filesystem>

namespace thirdai::search {

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
    raiseError(status, "open database");
  }

  if (handles.size() != 3) {
    throw std::runtime_error("Expected 3 handles to be created. Received " +
                             std::to_string(handles.size()) + " handles.");
  }

  _default = handles[0];
  _counters = handles[1];
  _token_to_docs = handles[2];

  updateNDocs(0);
  updateSumDocLens(0);
}

void RocksDbAdapter::storeDocLens(const std::vector<DocId>& ids,
                                  const std::vector<uint32_t>& doc_lens) {
  uint64_t sum_doc_lens = 0;

  rocksdb::Transaction* txn = _db->BeginTransaction(rocksdb::WriteOptions());

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const uint64_t doc_len = doc_lens[i];

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
      raiseError(merge_status, "add merge to batch");
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    raiseError(status, "write batch");
  }
}

void RocksDbAdapter::incrementDocLens(
    const std::vector<DocId>& ids,
    const std::vector<uint32_t>& doc_len_increments) {
  rocksdb::WriteBatch batch;

  uint64_t sum_new_lens = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    const uint64_t doc_len = doc_len_increments[i];
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

void RocksDbAdapter::incrementDocTokenCounts(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_doc_updates) {
  rocksdb::Transaction* txn = _db->BeginTransaction(rocksdb::WriteOptions());

  for (const auto& [token, updates] : token_to_doc_updates) {
    rocksdb::Slice key(reinterpret_cast<const char*>(&token), sizeof(token));

    std::string value;
    auto get_status =
        txn->GetForUpdate(rocksdb::ReadOptions(), _token_to_docs, key, &value);
    if (get_status.IsNotFound()) {
      TokenStatus status = TokenStatus::Default;
      value.append(reinterpret_cast<const char*>(&status), sizeof(TokenStatus));
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

std::vector<std::vector<DocCount>> RocksDbAdapter::lookupDocs(
    const std::vector<HashedToken>& query_tokens) const {
  std::vector<rocksdb::Slice> keys;
  keys.reserve(query_tokens.size());

  for (const auto& token : query_tokens) {
    keys.emplace_back(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  std::vector<rocksdb::PinnableSlice> values(keys.size());
  std::vector<rocksdb::Status> statuses(keys.size());

  _db->MultiGet(rocksdb::ReadOptions(), _token_to_docs, keys.size(),
                keys.data(), values.data(), statuses.data());

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
      raiseError(statuses[i], "batch get");
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

void RocksDbAdapter::removeDocs(const std::unordered_set<DocId>& docs) {
  rocksdb::Transaction* txn = _db->BeginTransaction(rocksdb::WriteOptions());

  for (const auto& doc : docs) {
    auto delete_status = txn->Delete(_counters, docIdKey(doc));
    if (!delete_status.ok()) {
      raiseError(delete_status, "txn delete");
    }
  }

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
      if (!docs.count(doc_counts[i].doc_id)) {
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

uint64_t RocksDbAdapter::getDocLen(DocId doc_id) const {
  std::string value;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _counters, docIdKey(doc_id), &value);
  if (!status.ok()) {
    raiseError(status, "get");
  }

  uint64_t len;
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
    raiseError(status, "get");
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
    raiseError(status, "get");
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
    raiseError(status, "merge");
  }
}

void RocksDbAdapter::updateSumDocLens(uint64_t sum_new_doc_lens) {
  auto status = _db->Merge(
      rocksdb::WriteOptions(), _counters, "sum_doc_lens",
      rocksdb::Slice(reinterpret_cast<const char*>(&sum_new_doc_lens),
                     sizeof(uint64_t)));
  if (!status.ok()) {
    raiseError(status, "merge");
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