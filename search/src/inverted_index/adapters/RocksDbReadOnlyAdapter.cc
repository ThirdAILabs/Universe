#include "RocksDbReadOnlyAdapter.h"
#include <rocksdb/merge_operator.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/write_batch.h>
#include <search/src/inverted_index/adapters/RocksDbUtils.h>
#include <filesystem>
#include <stdexcept>

namespace thirdai::search {

RocksDbReadOnlyAdapter::RocksDbReadOnlyAdapter(const std::string& save_path) {
  rocksdb::Options options;
  options.create_if_missing = false;
  options.create_missing_column_families = false;

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName,
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("counters",
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("token_to_docs",
                                      rocksdb::ColumnFamilyOptions()),
  };

  std::vector<rocksdb::ColumnFamilyHandle*> handles;

  auto status = rocksdb::DB::OpenForReadOnly(options, save_path,
                                             column_families, &handles, &_db);
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
}

void RocksDbReadOnlyAdapter::storeDocLens(
    const std::vector<DocId>& ids, const std::vector<uint32_t>& doc_lens) {
  (void)ids;
  (void)doc_lens;
  throw std::invalid_argument("This method is not supported for read only db.");
}

void RocksDbReadOnlyAdapter::updateTokenToDocs(
    const std::unordered_map<HashedToken, std::vector<DocCount>>&
        token_to_new_docs) {
  (void)token_to_new_docs;
  throw std::invalid_argument("This method is not supported for read only db.");
}

std::vector<std::vector<DocCount>> RocksDbReadOnlyAdapter::lookupDocs(
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

void RocksDbReadOnlyAdapter::prune(uint64_t max_docs_with_token) {
  (void)max_docs_with_token;
  throw std::invalid_argument("This method is not supported for read only db.");
}

void RocksDbReadOnlyAdapter::removeDocs(const std::unordered_set<DocId>& docs) {
  (void)docs;
  throw std::invalid_argument("This method is not supported for read only db.");
}

uint32_t RocksDbReadOnlyAdapter::getDocLen(DocId doc_id) const {
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

uint64_t RocksDbReadOnlyAdapter::getNDocs() const {
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

uint64_t RocksDbReadOnlyAdapter::getSumDocLens() const {
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

void RocksDbReadOnlyAdapter::save(const std::string& save_path) const {
  (void)save_path;
  throw std::invalid_argument("This method is not supported for read only db.");
}

RocksDbReadOnlyAdapter::~RocksDbReadOnlyAdapter() {
  _db->DestroyColumnFamilyHandle(_default);
  _db->DestroyColumnFamilyHandle(_counters);
  _db->DestroyColumnFamilyHandle(_token_to_docs);
  _db->Close();
  delete _db;
}

}  // namespace thirdai::search