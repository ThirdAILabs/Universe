#pragma once

#include <rocksdb/merge_operator.h>

namespace thirdai::search {

inline rocksdb::Slice asSlice(const uint64_t* value) {
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

}  // namespace thirdai::search