#pragma once

#include "../../utils/dataset/Dataset.h"
#include "../../utils/hashing/HashFunction.h"
#include "../../utils/hashtable/HashTable.h"

namespace thirdai::search {

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
  explicit Flash(const utils::HashFunction& function);

  /**
   * This is the same as the single argument constructor, except the supporting
   * hash table has a max reservoir size.
   **/
  Flash(const utils::HashFunction& function, uint32_t reservoir_size);

  /**
   * Insert all batches in the dataset the Flash data structure.
   * loadNextBatches on the dataset should not have been called yet, and this
   * will run through the entire dataset.
   */
  template <typename BATCH_T>
  void addDataset(utils::InMemoryDataset<BATCH_T>& dataset);

  template <typename BATCH_T>
  void addDataset(utils::StreamedDataset<BATCH_T>& dataset);

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

  ~Flash();

 private:
  /**
   * Returns a pointer to the hashes of the input batch. These hashes will need
   * to be deleted by the calling function.
   */
  template <typename BATCH_T>
  uint32_t* hash(const BATCH_T& batch) const;

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

  const utils::HashFunction& _function;
  const uint32_t _num_tables, _range;
  utils::HashTable<LABEL_T>* const _hashtable;
};

}  // namespace thirdai::search