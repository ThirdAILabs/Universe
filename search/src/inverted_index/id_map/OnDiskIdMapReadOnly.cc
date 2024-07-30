#include "OnDiskIdMapReadOnly.h"
#include <search/src/inverted_index/id_map/OnDiskUtils.h>
#include <stdexcept>

namespace thirdai::search {

OnDiskIdMapReadOnly::OnDiskIdMapReadOnly(const std::string& save_path) {
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

  auto status = rocksdb::DB::OpenForReadOnly(options, save_path,
                                             column_families, &handles, &_db);
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

std::vector<uint64_t> OnDiskIdMapReadOnly::get(uint64_t key) const {
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

uint64_t OnDiskIdMapReadOnly::maxKey() const {
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