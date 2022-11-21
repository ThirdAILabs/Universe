#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <hashing/src/HashFunction.h>
#include <hashtable/src/HashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <dataset/src/Datasets.h>

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
  explicit Flash(std::shared_ptr<hashing::HashFunction> hash_function);

  /**
   * This is the same as the single argument constructor, except the supporting
   * hash table has a max reservoir size.
   **/
  Flash(std::shared_ptr<hashing::HashFunction> hash_function,
        uint32_t reservoir_size);

  /**
   * This constructor SHOULD only be called when creating temporary Flash
   * objects to serialize into.
   */
  Flash<LABEL_T>() {}

  /**
   * Delete copy constructor and assignment operator
   */
  Flash& operator=(Flash&& flash_index) = delete;
  Flash(const Flash& flash_index) = delete;

  /**
   * Insert all batches in the dataset the Flash data structure.
   * loadNextBatches on the dataset should not have been called yet, and this
   * will run through the entire dataset.
   */
  void addDataset(const dataset::InMemoryDataset<BoltBatch>& dataset,
                  const std::vector<std::vector<LABEL_T>>& labels,
                  bool progress_bar);

  void addDataset(dataset::StreamingDataset<BoltBatch>& dataset,
                  const std::vector<std::vector<LABEL_T>>& labels);

  /**
   * Insert this batch into the Flash data structure.
   */
  void addBatch(const BoltBatch& batch, const std::vector<LABEL_T>& labels);

  /**
   * Perform a batch query on the Flash structure, for now on a Batch object.
   * If less than k results are found and pad_zeros = true, the results will be
   * padded with 0s to obtain a vector of length k. Otherwise less than k
   * results will be returned.
   */
  std::vector<std::vector<LABEL_T>> queryBatch(const BoltBatch& batch,
                                               uint32_t top_k,
                                               bool pad_zeros = false) const;

 private:
  /**
   * Returns a vector of concatenated hashes for the input batch
   */
  std::vector<uint32_t> hashBatch(const BoltBatch& batch) const;

  /**
   * Get the top_k labels that occur most often in the input vector using a
   * priority queue. The runtime of this method if O(nlogn) if query_result
   * has length n, because it must be sorted to find the top k. Note that
   * the input query_result will be modified (it will be sorted).
   */
  std::vector<LABEL_T> getTopKUsingPriorityQueue(
      std::vector<LABEL_T>& query_result, uint32_t top_k) const;

  std::shared_ptr<hashing::HashFunction> _hash_function;

  uint32_t _num_tables;
  uint32_t _range;

  std::shared_ptr<hashtable::HashTable<LABEL_T>> _hashtable;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hash_function, _num_tables, _range, _hashtable);
  }
};

}  // namespace thirdai::search
