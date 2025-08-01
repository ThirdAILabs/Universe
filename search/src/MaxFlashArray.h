#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlash.h"
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <dataset/src/InMemoryDataset.h>
#include <memory>
#include <utility>

namespace thirdai::search {

/**
 * Represents a collection of documents. We can incrementally update documents
 * and estimate the ColBERT score (sum of max similarities) between a query
 * and a document.
 * LABEL_T is an unsigned numerical type, currently uint8_t, uin16_t, uin32_t.
 * This class is NOT currently safe to call concurrently.
 */
template <typename LABEL_T>
class MaxFlashArray {
 public:
  // This will own the hash function and delete it during the destructor
  // Any documents passed in larger than max_doc_size or larger than
  // the max value of LABEL_T will be truncated.
  MaxFlashArray(hashing::HashFunction* function, uint32_t hashes_per_table,
                uint64_t max_doc_size = std::numeric_limits<LABEL_T>::max());

  // This needs to be public since it's a top level serialization target, but
  // DO NOT call it unless you are creating a temporary object to serialize
  // into.
  MaxFlashArray(){};

  uint64_t addDocument(const BoltBatch& batch);

  std::vector<float> getDocumentScores(
      const BoltBatch& query,
      const std::vector<uint32_t>& documents_to_query) const;

  // Delete copy constructor and assignment
  MaxFlashArray(const MaxFlashArray&) = delete;
  MaxFlashArray& operator=(const MaxFlashArray&) = delete;

 private:
  template <typename BATCH_T>
  std::vector<uint32_t> hash(const BATCH_T& batch) const;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_max_allowable_doc_size, _hash_function, _maxflash_array,
            _collision_count_to_sim);
  }

  LABEL_T _max_allowable_doc_size;
  std::unique_ptr<hashing::HashFunction> _hash_function;
  std::vector<std::unique_ptr<MaxFlash<LABEL_T>>> _maxflash_array;
  std::vector<float> _collision_count_to_sim;
};

}  // namespace thirdai::search