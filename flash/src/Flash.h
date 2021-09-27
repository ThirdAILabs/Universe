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
template <typename Label_t>
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
   * Insert all batches in the dataset the Flash data structure.
   * loadNextBatches on the dataset should not have been called yet, and this
   * will run through the entire dataset.
   */
  void addDataset(utils::Dataset& dataset);

  /** Insert this batch into the Flash data structure. */
  void addBatch(const utils::Batch& batch);

  /**
   * Perform a batch query on the Flash structure, for now on a Batch object.
   * Note that if not enough results are found, the results will be padded
   * with 0s.
   */
  std::vector<std::vector<Label_t>> queryBatch(const utils::Batch& batch,
                                               uint32_t top_k) const;

 private:
  /**
   * Returns a pointer to the hashes of the input batch. These hashes will need
   * to be deleted by the calling function.
   */
  uint32_t* hash(const utils::Batch& batch) const;

  /**
   * Get the top_k labels that occur most often in the input vector using a
   * priority queue. The runtime of this method if O(nlogn) if query_result
   * has length n, because it must be sorted to find the top k. Note that
   * the input query_result will be modified (it will be sorted).
   */
  std::vector<Label_t> getTopKUsingPriorityQueue(
      std::vector<Label_t>& query_result, uint32_t top_k) const;

  /**
   * Throws an error if this batch contains an id that is too large for the
   * initialized FLASH (>2^16 for uin16_t, >2^32 for uint32_t, etc.).
   */
  void verifyBatchIds(const utils::Batch& batch) const;

  const utils::HashFunction& _function;
  uint32_t _num_tables, _range;
  utils::HashTable<Label_t>* _hashtable;
};

}  // namespace thirdai::search