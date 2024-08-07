#include "OnDiskIdMap.h"
#include <search/src/inverted_index/id_map/OnDiskUtils.h>
#include <stdexcept>

namespace thirdai::search {

OnDiskIdMap::OnDiskIdMap(const std::string& save_path, bool read_only)
    : _save_path(save_path) {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  rocksdb::ColumnFamilyOptions value_to_keys_options;
  value_to_keys_options.merge_operator = std::make_shared<Append>();

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName,
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("value_to_keys", value_to_keys_options),
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
    throw std::runtime_error(open_status.ToString() + "open");
  }

  if (handles.size() != 2) {
    throw std::runtime_error("Expected 2 handles to be created. Received " +
                             std::to_string(handles.size()) + " handles.");
  }

  _key_to_values = handles[0];
  _value_to_keys = handles[1];
}

std::vector<uint64_t> OnDiskIdMap::get(uint64_t key) const {
  std::string value;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _key_to_values, asSlice(&key), &value);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString() + "get");
  }

  if (value.size() % sizeof(uint64_t) != 0) {
    throw std::runtime_error("Value corruption");
  }

  const uint64_t* ptr = reinterpret_cast<const uint64_t*>(value.data());
  return {ptr, ptr + value.size() / sizeof(uint64_t)};
}

void OnDiskIdMap::put(uint64_t key, const std::vector<uint64_t>& values) {
  rocksdb::WriteBatch batch;

  auto forward_status = batch.Put(_key_to_values, asSlice(&key),
                                  {reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(uint64_t)});

  if (!forward_status.ok()) {
    throw std::runtime_error(forward_status.ToString() + "add put to batch");
  }

  for (const auto& value : values) {
    auto reverse_status =
        batch.Merge(_value_to_keys, asSlice(&value), asSlice(&key));

    if (!reverse_status.ok()) {
      throw std::runtime_error(reverse_status.ToString() +
                               "add merge to batch");
    }
  }

  auto status = _db->Write(rocksdb::WriteOptions(), &batch);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString() + "write batch");
  }
}

std::vector<uint64_t> OnDiskIdMap::deleteValue(uint64_t value) {
  auto* txn = startTransaction();

  std::string serialized_keys;
  auto reverse_status =
      txn->GetForUpdate(rocksdb::ReadOptions(), _value_to_keys, asSlice(&value),
                        &serialized_keys);
  if (reverse_status.IsNotFound()) {
    delete txn;

    return {};
  }

  if (!reverse_status.ok()) {
    throw std::runtime_error(reverse_status.ToString() + "txn get");
  }

  std::vector<uint64_t> empty_keys;

  const uint64_t* keys =
      reinterpret_cast<const uint64_t*>(serialized_keys.data());
  if (serialized_keys.size() % sizeof(uint64_t) != 0) {
    throw std::runtime_error("Data corruption");
  }
  const size_t n_keys = serialized_keys.size() / sizeof(uint64_t);

  for (size_t i = 0; i < n_keys; i++) {
    const uint64_t key = keys[i];
    std::string serialized_values;
    auto get_status = txn->GetForUpdate(rocksdb::ReadOptions(), asSlice(&key),
                                        &serialized_values);
    if (!get_status.ok()) {
      throw std::runtime_error(get_status.ToString() + "txn get");
    }

    uint64_t* values = reinterpret_cast<uint64_t*>(serialized_values.data());
    if (serialized_values.size() % sizeof(uint64_t) != 0) {
      throw std::runtime_error("Data corruption");
    }
    size_t n_values = serialized_values.size() / sizeof(uint64_t);

    auto* loc = std::find(values, values + n_values, value);
    if (loc != values + n_values) {
      *loc = values[n_values - 1];
      n_values--;
    }
    if (n_values == 0) {
      auto del_status = txn->Delete(_key_to_values, asSlice(&key));
      if (!del_status.ok()) {
        throw std::runtime_error(del_status.ToString() + "txn delete");
      }
      empty_keys.push_back(key);
    } else {
      auto put_status = txn->Put(
          _key_to_values, asSlice(&key),
          {reinterpret_cast<const char*>(values), n_values * sizeof(uint64_t)});
      if (!put_status.ok()) {
        throw std::runtime_error(put_status.ToString() + "txn put");
      }
    }
  }

  auto del_status = txn->Delete(_value_to_keys, asSlice(&value));
  if (!del_status.ok()) {
    throw std::runtime_error(del_status.ToString() + "txn delete");
  }

  auto status = txn->Commit();
  if (!status.ok()) {
    throw std::runtime_error(status.ToString() + "txn commit");
  }

  delete txn;

  return empty_keys;
}

uint64_t OnDiskIdMap::maxKey() const {
  auto* iter = _db->NewIterator(rocksdb::ReadOptions());

  uint64_t max_key = 0;
  for (iter->SeekToFirst(); iter->Valid(); iter->Next()) {
    uint64_t key = *reinterpret_cast<const uint64_t*>(iter->key().data());
    if (key > max_key) {
      max_key = key;
    }
  }

  delete iter;

  return max_key;
}

}  // namespace thirdai::search