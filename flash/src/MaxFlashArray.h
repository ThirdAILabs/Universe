#pragma once

#include "MaxFlash.h"
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <dataset/src/Dataset.h>
#include <memory>
#include <utility>

namespace thirdai::search {

template <typename LABEL_T>
class MaxFlashArray {
 public:
  // This will own the hash function and delete it during the destructor
  MaxFlashArray(hashing::HashFunction* function, uint64_t num_flashes,
                uint32_t hashes_per_table);

  template <typename BATCH_T>
  void addDocument(const BATCH_T& batch, uint64_t document_id);

  std::vector<float> getDocumentScores(
      const dataset::DenseBatch& query,
      const std::vector<uint32_t>& documents_to_query) const;

  ~MaxFlashArray();

 private:
  /**
   * Returns a pointer to the hashes of the input batch. These hashes will need
   * to be deleted by the calling function.
   */

  template <typename BATCH_T>
  uint32_t* hash(const BATCH_T& batch) const;

  hashing::HashFunction* _function;
  std::vector<MaxFlash<LABEL_T>*> _maxflash_array;
  std::vector<float> _lookups;
  uint32_t _largest_doc;
};

}  // namespace thirdai::search