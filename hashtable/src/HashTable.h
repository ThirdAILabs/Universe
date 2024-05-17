#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

/**
 * This is the abstract HashTable interface, which represents n hash tables
 * with a certain range. It abstracts batch indexing and single queries.
 * Implementations are intended to parallilize batch indexing.
 */
template <typename LABEL_T>
class HashTable {
 public:
  /**
   * Do a parallel insertion of n elements. The hashes should be in hash major
   * order, i.e. as returned by the HashFunction.h class, the ith hash of the
   * jth vector should be in position num_tables * i + j. The label of the
   * ith vector to insert into the hash table should be labels[i]. All hashes
   * should be less than the hash table range, and there should be exactly
   * num_tables * n number of them. This will not be checked and will cause
   * segfaults if it is not followed.
   */
  virtual void insert(uint64_t n, LABEL_T const* labels,
                      uint32_t const* hashes) = 0;

  /**
   * Same as the insert method, except the ith vector will be inserted with
   * the label start + i.
   */
  virtual void insertSequential(uint64_t n, LABEL_T start,
                                uint32_t const* hashes) = 0;

  /**
   * The hashes array should have length equal to the number of tables, and
   * the ith entry should be a bucket index into the ith table. This query
   * adds to the store set all labels that are in any of the hashed to buckets
   * across all tables.
   */
  virtual void queryBySet(uint32_t const* hashes,
                          std::unordered_set<LABEL_T>& store) const = 0;

  virtual void queryByCount(uint32_t const* hashes,
                            std::vector<uint32_t>& counts) const = 0;

  /**
   * Same as queryBySet, except adds to the results vector all labels that are
   * in any of the hashed to buckets. A label can appear twice in the results
   * vector if it was in multiple buckets.
   */
  virtual void queryByVector(uint32_t const* hashes,
                             std::vector<LABEL_T>& results) const = 0;

  /** Removes all elements from all tables */
  virtual void clearTables() = 0;

  /** Returns the total number of tables */
  virtual uint32_t numTables() const = 0;

  /* Returns the range (number of buckets) of each table */
  virtual uint64_t tableRange() const = 0;

  virtual ~HashTable() {}

 protected:
  /** The default number of random pregenerated numbers to use for sampling */
  const static uint32_t DEFAULT_MAX_RAND = 10000;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

template <typename LABEL_T>
const uint32_t HashTable<LABEL_T>::DEFAULT_MAX_RAND;

}  // namespace thirdai::hashtable