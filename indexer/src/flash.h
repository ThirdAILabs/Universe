#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <dataset/src/Datasets.h>

namespace thirdai::automl::deployment {

/**
 * See https://arxiv.org/pdf/2106.11565.pdf for the original Flash paper.
 * The template parameter for this class represents the largest label that
 * will be inserted, and if a label larger than this is inserted the insert
 * method will throw an error.
 */
template <typename LABEL_T>
class Flash {
 public:
  /**
   * Construct Flash with a given HashFunction. Note that this will use the
   * input HashFunction's numTables and range functions to construct an
   * internal HashTable, so make sure that the range is what you want (you
   * may have to mod it and change the range, or do that in the hashfunction
   * implementation).
   **/
  explicit Flash(hashing::HashFunction* function);

  /**
   * This is the same as the single argument constructor, except the supporting
   * hash table has a max reservoir size.
   **/
  Flash(hashing::HashFunction* function, uint32_t reservoir_size);

  /* Constructor called when creating temporary Flash objects to serialize
   * into */
  Flash<LABEL_T>() {}

  // Flash& operator=(Flash&& flash_index) = default;

  /**
   * Insert all batches in the dataset the Flash data structure.
   * loadNextBatches on the dataset should not have been called yet, and this
   * will run through the entire dataset.
   */
  template <typename BATCH_T>
  void addDataset(dataset::InMemoryDataset<BATCH_T>& dataset);

  template <typename BATCH_T>
  void addDataset(dataset::StreamingDataset<BATCH_T>& dataset);

  /** Insert this batch into the Flash data structure. */
  template <typename BATCH_T>
  void addBatch(const BATCH_T& batch);

  /**
   * Perform a batch query on the Flash structure, for now on a Batch object.
   * If less than k results are found and pad_zeros = true, the results will be
   * padded with 0s to obtain a vector of length k. Otherwise less than k
   * results will be returned.
   */
  template <typename BATCH_T>
  std::vector<std::vector<LABEL_T>> queryBatch(const BATCH_T& batch,
                                               uint32_t top_k,
                                               bool pad_zeros = false) const;

 private:
  /**
   * Returns a vector of hashes for the input batch
   */
  template <typename BATCH_T>
  std::vector<uint32_t> hash_batch(const BATCH_T& batch) const;

  /**
   * Get the top_k labels that occur most often in the input vector using a
   * priority queue. The runtime of this method if O(nlogn) if query_result
   * has length n, because it must be sorted to find the top k. Note that
   * the input query_result will be modified (it will be sorted).
   */
  std::vector<LABEL_T> getTopKUsingPriorityQueue(
      std::vector<LABEL_T>& query_result, uint32_t top_k) const;

  /** Makes sure the ids are within range for a batch with sequential ids */
  template <typename BATCH_T>
  void verifyBatchSequentialIds(const BATCH_T& batch) const;

  /**
   * Verifies that the passed in id is within the range of this FLASH instance
   * by throwing an error if the id is too large for the initialized size
   * (>2^16 for uin16_t, >2^32 for uint32_t, etc.).
   */
  LABEL_T verify_and_convert_id(uint64_t id) const;

  std::unique_ptr<hashing::HashFunction> _hash_function;

  uint32_t _num_tables;
  uint32_t _range;
  std::shared_ptr<hashtable::HashTable<LABEL_T>> _hashtable;

  // Handle serialization
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _num_tables, _range, _hashtable);
  }
};

}  // namespace thirdai::automl::deployment