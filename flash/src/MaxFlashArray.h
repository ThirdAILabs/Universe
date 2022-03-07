#pragma once

#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include "MaxFlash.h"
#include <hashing/src/FastSRP.h>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <dataset/src/Dataset.h>
#include <memory>
#include <utility>

namespace thirdai::search {

// TODO(josh): This class is NOT currently safe to call concurrently.
// Fix this.
template <typename LABEL_T>
class MaxFlashArray {
 public:
  // This will own the hash function and delete it during the destructor
  // TODO(josh): Remove naked pointers from hash function library so moves will
  // work, then change this to a unique pointer.
  // Any documents passed in larger than max_doc_size or larger than
  // the max value of LABEL_T will be truncated.
  // TODO(josh): Change truncation to error?
  // TODO(josh): Make LABEL_T allowed to be different for each document, so
  // it is as space efficient as possible
  MaxFlashArray(hashing::HashFunction* function, uint32_t hashes_per_table,
                uint64_t max_doc_size);

  MaxFlashArray(){};

  template <typename BATCH_T>
  uint64_t addDocument(const BATCH_T& batch);

  std::vector<float> getDocumentScores(
      const dataset::DenseBatch& query,
      const std::vector<uint32_t>& documents_to_query) const;

  // Delete copy constructor and assignment
  MaxFlashArray(const MaxFlashArray&) = delete;
  MaxFlashArray& operator=(const MaxFlashArray&) = delete;

  // This method lets cereal know which data members to serialize
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_max_allowable_doc_size, _maxflash_array, _collision_count_to_sim);
  }

 private:
  template <typename BATCH_T>
  std::vector<uint32_t> hash(const BATCH_T& batch) const;

  // MaxFlashArray<LABEL_T>() {}
  // friend class cereal::access;

  LABEL_T _max_allowable_doc_size;
  std::unique_ptr<hashing::HashFunction> _hash_function;
  std::vector<std::unique_ptr<MaxFlash<LABEL_T>>> _maxflash_array;
  std::vector<float> _collision_count_to_sim;
};

}  // namespace thirdai::search