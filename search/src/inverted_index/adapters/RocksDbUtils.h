#pragma once

#include <rocksdb/merge_operator.h>
#include <rocksdb/slice.h>
#include <rocksdb/status.h>
#include <search/src/inverted_index/DbAdapter.h>
#include <search/src/inverted_index/Retriever.h>

namespace thirdai::search {

inline void raiseError(const rocksdb::Status& status, const std::string& op) {
  throw std::runtime_error(status.ToString() + op);
}

inline std::string docIdKey(DocId doc_id) {
  // We are prependeding the letter D to the keys so that we could do a prefix
  // scan to iterate over the different doc_ids in the database if needed. We
  // are serializing the numbers instead of converting to a string because it
  // will be more space efficient at larger scale.
  std::string key;
  key.reserve(sizeof(DocId) + 1);
  key.append("D");
  key.append(reinterpret_cast<const char*>(&doc_id), sizeof(DocId));
  return key;
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

inline bool isPruned(const rocksdb::Slice& value) {
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

inline DocCount* docCountPtr(std::string& value) {
  return reinterpret_cast<DocCount*>(value.data() + sizeof(TokenStatus));
}

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class AppendDocCounts : public rocksdb::AssociativeMergeOperator {
  /**
   * This merge operator appends a list of serialized DocCounts to another
   * existing list. At the begining a TokenStatus is stored, this is to act as a
   * tombstone in case the token is pruned. Without the token status we would
   * have to worry about a token being pruned, then added back in a future doc,
   * having the status allows us to distinguish between tokens that were pruned,
   * and tokens that were not yet seen by the index.
   */
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    if (!existing_value) {
      *new_value = std::string(sizeof(TokenStatus) + value.size(), 0);

      *reinterpret_cast<TokenStatus*>(new_value->data()) = TokenStatus::Default;

      std::copy(value.data(), value.data() + value.size(),
                new_value->data() + sizeof(TokenStatus));

      return true;
    }

    if (isPruned(*existing_value)) {
      *new_value = existing_value->ToString();
      return true;
    }

    // Note: this assumes that the doc_ids in the old and new values are
    // disjoint. This is true because we check for duplicate doc_ids during
    // indexing. If we add support for updates, then this needs to be modified
    // to merge the 2 values based on doc_id.

    *new_value = std::string();
    new_value->reserve(existing_value->size() + value.size());

    new_value->append(existing_value->data(), existing_value->size());
    new_value->append(value.data(), value.size());

    return true;
  }

  const char* Name() const override { return "AppendDocTokenCount"; }
};

class IncrementCounter : public rocksdb::AssociativeMergeOperator {
  /**
   * This merge operator is a simple counter operator, that will add the new
   * value to the existing value.
   */
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    int64_t counter = 0;
    if (existing_value) {
      if (!deserialize(*existing_value, counter)) {
        return false;
      }
    }

    int64_t increment = 0;
    if (!deserialize(value, increment)) {
      return false;
    }

    *new_value = std::string(sizeof(int64_t), 0);
    *reinterpret_cast<int64_t*>(new_value->data()) = counter + increment;

    return true;
  }

  const char* Name() const override { return "IncrementCounter"; }
};

}  // namespace thirdai::search