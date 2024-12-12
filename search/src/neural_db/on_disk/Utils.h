#pragma once

#include <rocksdb/slice.h>
#include <rocksdb/utilities/transaction.h>

namespace thirdai::search::ndb {

using TxnPtr = std::unique_ptr<rocksdb::Transaction>;

template <typename T>
inline rocksdb::Slice asSlice(const T* item) {
  return rocksdb::Slice(reinterpret_cast<const char*>(item), sizeof(T));
}

}  // namespace thirdai::search::ndb