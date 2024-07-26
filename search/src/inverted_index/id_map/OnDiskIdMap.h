#pragma once

#include <rocksdb/db.h>
#include <rocksdb/utilities/transaction_db.h>
#include <search/src/inverted_index/id_map/IdMap.h>
#include <filesystem>
#include <memory>
#include <string>

namespace thirdai::search {

class OnDiskIdMap final : public IdMap {
 public:
  explicit OnDiskIdMap(const std::string& save_path);

  std::vector<uint64_t> get(uint64_t key) const final;

  void put(uint64_t key, const std::vector<uint64_t>& values) final;

  std::vector<uint64_t> deleteValue(uint64_t value) final;

  void save(const std::string& save_path) const final {
    std::filesystem::copy(_save_path, save_path,
                          std::filesystem::copy_options::recursive);
  }

  uint64_t maxKey() const final;

  static std::unique_ptr<OnDiskIdMap> load(const std::string& save_path) {
    return std::make_unique<OnDiskIdMap>(save_path);
  }

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "on-disk"; }

  ~OnDiskIdMap() final {
    _db->DestroyColumnFamilyHandle(_forward);
    _db->DestroyColumnFamilyHandle(_reverse);
    _db->Close();
    delete _db;
  }

 private:
  static inline rocksdb::Slice asSlice(const uint64_t* value) {
    return {reinterpret_cast<const char*>(value), sizeof(uint64_t)};
  }

  rocksdb::TransactionDB* _db;

  rocksdb::ColumnFamilyHandle* _forward;
  rocksdb::ColumnFamilyHandle* _reverse;

  std::string _save_path;
};

}  // namespace thirdai::search