#pragma once

#include <rocksdb/db.h>
#include <rocksdb/merge_operator.h>
#include <search/src/inverted_index/IdMap.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::search {

class OnDiskIdMap final : public IdMap {
 public:
  explicit OnDiskIdMap(const std::string& save_path) : _save_path(save_path) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.merge_operator = std::make_shared<Append>();

    auto status = rocksdb::DB::Open(options, save_path, &_db);
    if (!status.ok()) {
      throw std::runtime_error(status.ToString() + "open");
    }
  }

  std::vector<uint64_t> get(uint64_t key) const final {
    std::string value;
    auto status = _db->Get(rocksdb::ReadOptions(), asSlice(&key), &value);
    if (!status.ok()) {
      throw std::runtime_error(status.ToString() + "get");
    }

    if (value.size() % sizeof(uint64_t) != 0) {
      throw std::runtime_error("Value corruption");
    }

    const uint64_t* ptr = reinterpret_cast<const uint64_t*>(value.data());
    return {ptr, ptr + value.size() / sizeof(uint64_t)};
  }

  bool contains(uint64_t key) const final {
    std::string value;
    auto status = _db->Get(rocksdb::ReadOptions(), asSlice(&key), &value);
    if (status.ok()) {
      return true;
    }
    if (status.IsNotFound()) {
      return false;
    }
    throw std::runtime_error(status.ToString() + "get");
  }

  void put(uint64_t key, std::vector<uint64_t> value) final {
    auto status = _db->Put(rocksdb::WriteOptions(), asSlice(&key),
                           {reinterpret_cast<const char*>(value.data()),
                            value.size() * sizeof(uint64_t)});
    if (!status.ok()) {
      throw std::runtime_error(status.ToString() + "put");
    }
  }

  void append(uint64_t key, uint64_t value) final {
    auto status =
        _db->Merge(rocksdb::WriteOptions(), asSlice(&key), asSlice(&value));
    if (!status.ok()) {
      throw std::runtime_error(status.ToString() + "merge");
    }
  }

  void del(uint64_t key) final {
    auto status = _db->Delete(rocksdb::WriteOptions(), asSlice(&key));
    if (!status.ok()) {
      throw std::runtime_error(status.ToString() + "delete");
    }
  }

  void save(const std::string& save_path) const final {
    std::filesystem::copy(_save_path, save_path,
                          std::filesystem::copy_options::recursive);
  }

  static std::unique_ptr<OnDiskIdMap> load(const std::string& save_path) {
    return std::make_unique<OnDiskIdMap>(save_path);
  }

  std::string type() const final { return typeName(); }

  static std::string typeName() { return "on-disk"; }

  ~OnDiskIdMap() final {
    _db->Close();
    delete _db;
  }

 private:
  static inline rocksdb::Slice asSlice(const uint64_t* value) {
    return {reinterpret_cast<const char*>(value), sizeof(uint64_t)};
  }

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

  rocksdb::DB* _db;
  std::string _save_path;
};

}  // namespace thirdai::search