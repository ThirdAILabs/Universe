#pragma once

#include <rocksdb/merge_operator.h>
#include <search/src/neural_db/on_disk/ChunkCountView.h>

namespace thirdai::search::ndb {

// https://github.com/facebook/rocksdb/wiki/Merge-Operator
class ConcatChunkCounts : public rocksdb::AssociativeMergeOperator {
  /**
   * This merge operator appends a list of serialized ChunkCounts to another
   * existing list. Inputs are an existing_value and value are a serialized
   * array. The existing_value arg is the current value associated with the key,
   * whereas the value arg indicates the new ChunkCounts that are being
   * appended. The output is a new value in the same format as the inputs, only
   * with the ChunkCounts from each the input values concatenated.
   *
   * If the token has been pruned, then the value will be a single ChunkCount
   * with all it's bits set to 1. This acts as a tombstone for the token.
   * Without the tombstone we would have to worry about a token being pruned,
   * then added back in a future chunk, having the status allows us to
   * distinguish between tokens that were pruned, and tokens that were not yet
   * seen by the index.
   *
   *
   * Example:
   *
   * During indexing a batch of chunks token T occurs in chunks 0 and 1 with
   * counts of 10 and 11 respectively. Essentially this will result in a merge
   * like this:
   *    Merge(
   *      existing=[],
          update=[(chunk=1, cnt=10), (chunk=2, cnt=11)]
   *    ) -> [(chunk=0, cnt=10), (chunk=1, cnt=11)]
   *
   * Later, indexing more chunks with token T occuring in chunk 2 with count 12
   * will result in a merge like this:
   *    Merge(
   *      existing=[(chunk=1, cnt=10), (chunk=2, cnt=11)],
   *      update=[(chunk=2, cnt=12)]
   *    ) -> [(chunk=0, cnt=10), (chunk=1, cnt=11), (chunk=2, cnt=12)]
   *
   */
 public:
  bool Merge(const rocksdb::Slice& key, const rocksdb::Slice* existing_value,
             const rocksdb::Slice& value, std::string* new_value,
             rocksdb::Logger* logger) const override {
    (void)key;
    (void)logger;

    if (!existing_value) {
      *new_value = value.ToString();
      return true;
    }

    ChunkCountView existing_value_view(*existing_value);
    ChunkCountView value_view(value);

    if (existing_value_view.isPruned()) {
      *new_value = existing_value->ToString();
      return true;
    }

    if (value_view.isPruned()) {
      *new_value = value.ToString();
      return true;
    }

    // Note: this assumes that the chunk_ids in the old and new values are
    // disjoint. This is true because each insert creates new chunk_ids. If we
    // want to support chunk updates, then this needs to be modified to merge
    // the 2 values based on chunk_id.
    *new_value = std::string();
    new_value->reserve(existing_value->size() + value.size());

    new_value->append(existing_value->data(), existing_value->size());
    new_value->append(value.data(), value.size());

    return true;
  }

  const char* Name() const override { return "ConcatChunkCounts"; }
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

 private:
  static bool deserialize(const rocksdb::Slice& value, int64_t& counter) {
    if (value.size() != sizeof(int64_t)) {
      return false;
    }
    counter = *reinterpret_cast<const int64_t*>(value.data());
    return true;
  }
};

}  // namespace thirdai::search::ndb