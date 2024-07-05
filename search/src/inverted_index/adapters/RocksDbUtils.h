#pragma once

#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <search/src/inverted_index/DbAdapter.h>

namespace thirdai::search {

inline void raiseError(const std::string& op, const rocksdb::Status& status) {
  throw std::runtime_error(op + " failed with error: " + status.ToString() +
                           ".");
}

inline std::string docIdKey(uint64_t doc_id) {
  return "doc_" + std::to_string(doc_id);
}

// Using uint32_t since this will be prepended to doc counts, and so uint32_t
// ensures that it is still half-word aligned.
enum class TokenStatus : uint32_t {
  Default = 0,
  Pruned = 1,
};

template <typename T>
inline bool deserialize(const rocksdb::Slice& value, T& output) {
  if (value.size() != sizeof(T)) {
    return false;
  }
  output = *reinterpret_cast<const T*>(value.data());
  return true;
}

inline bool isPruned(const std::string& value) {
  auto status = *reinterpret_cast<const TokenStatus*>(value.data());
  return status == TokenStatus::Pruned;
}

inline size_t docsWithToken(const rocksdb::Slice& value) {
  assert((value.size() - sizeof(TokenStatus)) % sizeof(DocCount) == 0);
  return (value.size() - sizeof(TokenStatus)) / sizeof(DocCount);
}

inline const DocCount* docCountPtr(const rocksdb::Slice& value) {
  return reinterpret_cast<const DocCount*>(value.data() + sizeof(TokenStatus));
}

}  // namespace thirdai::search