#include "OnDiskIdMap.h"
#include <rocksdb/merge_operator.h>
#include <stdexcept>

namespace thirdai::search {

class Append : public rocksdb::AssociativeMergeOperator {
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    if (existing_value) {
      *new_value = std::string();
      new_value->reserve(existing_value->size() + value.size());
      new_value->append(existing_value->data(), existing_value->size());
      new_value->append(value.data(), value.size());
    } else {
      *new_value = value.ToString();
    }

    return true;
  }

  const char* Name() const override { return "Append"; }
};

OnDiskIdMap::OnDiskIdMap(const std::string& save_path) : _save_path(save_path) {
  rocksdb::Options options;
  options.create_if_missing = true;
  options.create_missing_column_families = true;

  rocksdb::ColumnFamilyOptions reverse_options;
  reverse_options.merge_operator = std::make_shared<Append>();

  std::vector<rocksdb::ColumnFamilyDescriptor> column_families = {
      rocksdb::ColumnFamilyDescriptor(rocksdb::kDefaultColumnFamilyName,
                                      rocksdb::ColumnFamilyOptions()),
      rocksdb::ColumnFamilyDescriptor("reverse", reverse_options),
  };

  std::vector<rocksdb::ColumnFamilyHandle*> handles;

  auto status =
      rocksdb::TransactionDB::Open(options, rocksdb::TransactionDBOptions(),
                                   save_path, column_families, &handles, &_db);
  if (!status.ok()) {
    throw std::runtime_error(status.ToString() + "open");
  }

  if (handles.size() != 2) {
    throw std::runtime_error("Expected 2 handles to be created. Received " +
                             std::to_string(handles.size()) + " handles.");
  }

  _forward = handles[0];
  _reverse = handles[1];
}

std::vector<uint64_t> OnDiskIdMap::get(uint64_t key) const {
  std::string value;
  auto status =
      _db->Get(rocksdb::ReadOptions(), _forward, asSlice(&key), &value);
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

  auto forward_status = batch.Put(_forward, asSlice(&key),
                                  {reinterpret_cast<const char*>(values.data()),
                                   values.size() * sizeof(uint64_t)});

  if (!forward_status.ok()) {
    throw std::runtime_error(forward_status.ToString() + "add put to batch");
  }

  for (const auto& value : values) {
    auto reverse_status = batch.Merge(_reverse, asSlice(&value), asSlice(&key));

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
  auto* txn = _db->BeginTransaction(rocksdb::WriteOptions());

  std::string serialized_keys;
  auto reverse_status = txn->GetForUpdate(rocksdb::ReadOptions(), _reverse,
                                          asSlice(&value), &serialized_keys);
  if (reverse_status.IsNotFound()) {
    return {};
  }

  if (!reverse_status.ok()) {
    throw std::runtime_error(reverse_status.ToString() + "txn get");
  }

  std::vector<uint64_t> empty_keys;

  const uint64_t* keys =
      reinterpret_cast<const uint64_t*>(serialized_keys.data());
  // TODO(Nicholas): check size is multiple
  const size_t n_keys = serialized_keys.size() / sizeof(uint64_t);

  for (size_t i = 0; i < n_keys; i++) {
    const uint64_t key = keys[i];
    std::string serialized_values;
    txn->GetForUpdate(rocksdb::ReadOptions(), asSlice(&key),
                      &serialized_values);

    uint64_t* values = reinterpret_cast<uint64_t*>(serialized_values.data());
    // TODO(Nicholas): check size is multiple
    size_t n_values = serialized_values.size() / sizeof(uint64_t);

    auto* loc = std::find(values, values + n_values, value);
    if (loc != values + n_values) {
      *loc = values[n_values - 1];
      n_values--;
    }
    if (n_values == 0) {
      txn->Delete(_forward, asSlice(&key));
      empty_keys.push_back(key);
    } else {
      txn->Put(
          _forward, asSlice(&key),
          {reinterpret_cast<const char*>(values), n_values * sizeof(uint64_t)});
    }
  }

  txn->Delete(_reverse, asSlice(&value));

  txn->Commit();

  delete txn;

  return empty_keys;
}

}  // namespace thirdai::search