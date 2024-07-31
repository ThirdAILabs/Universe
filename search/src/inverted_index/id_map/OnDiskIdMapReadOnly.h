#pragma once

#include <rocksdb/db.h>
#include <search/src/inverted_index/id_map/IdMap.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::search {

class OnDiskIdMapReadOnly final : public IdMap {
 public:
  explicit OnDiskIdMapReadOnly(const std::string& save_path);

  std::vector<uint64_t> get(uint64_t key) const final;

  void put(uint64_t key, const std::vector<uint64_t>& values) final {
    (void)key;
    (void)values;
    throw std::invalid_argument(
        "This method is not supported in read only mode.");
  }

  std::vector<uint64_t> deleteValue(uint64_t value) final {
    (void)value;
    throw std::invalid_argument(
        "This method is not supported in read only mode.");
  }

  uint64_t maxKey() const final;

  void save(const std::string& save_path) const final {
    (void)save_path;
    throw std::invalid_argument(
        "This method is not supported in read only mode.");
  }

  static std::unique_ptr<OnDiskIdMapReadOnly> load(
      const std::string& save_path) {
    return std::make_unique<OnDiskIdMapReadOnly>(save_path);
  }

  std::string type() const final {
    throw std::invalid_argument(
        "This method is not supported in read only mode.");
  }

  ~OnDiskIdMapReadOnly() final {
    _db->DestroyColumnFamilyHandle(_key_to_values);
    _db->DestroyColumnFamilyHandle(_value_to_keys);
    _db->Close();
    delete _db;
  }

 private:
  rocksdb::DB* _db;

  rocksdb::ColumnFamilyHandle* _key_to_values;
  rocksdb::ColumnFamilyHandle* _value_to_keys;
};

}  // namespace thirdai::search