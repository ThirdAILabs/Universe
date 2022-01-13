#pragma once

#include "HashTable.h"
#include <atomic>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

/**
 * This class implements the hash table interface for a sampled hash table where
 * reservoirs are inserted into via the reservoir sampling algorithm when full.
 *
 * This allows for reservoirs to be preallocated and makes insertion and
 * querying fast.
 */
template <typename LABEL_T>
class SampledHashTable final : public HashTable<LABEL_T> {
 private:
  uint64_t _num_tables, _reservoir_size, _range, _max_rand;

  LABEL_T* _data;
  std::atomic<uint32_t>* _counters;

  uint32_t* _gen_rand;

  constexpr uint64_t CounterIdx(uint64_t table, uint64_t row) const {
    return table * _range + row;
  }

  constexpr uint64_t DataIdx(uint64_t table, uint64_t row,
                             uint64_t offset) const {
    return table * _range * _reservoir_size + row * _reservoir_size + offset;
  }

  /** Helper method that inserts a given label into the hash tables */
  void insertIntoTables(LABEL_T label, const uint32_t* hashes);

 public:
  /**
   * num_tables: number of hash tables
   * reservoir_size: the size of each reservoir
   * range_pow: log base 2 of the range of the table
   * seed: optional parameter, controls seed of the pre-generated random vals
   * max_rand: optional parameter, controls how many pre-generated random values
   * are created for reservoir sampling
   */
  SampledHashTable(uint64_t num_tables, uint64_t reservoir_size, uint64_t range,
                   uint32_t seed = time(nullptr),
                   uint64_t max_rand = HashTable<LABEL_T>::DEFAULT_MAX_RAND);

  SampledHashTable(const SampledHashTable& other) = delete;

  SampledHashTable& operator=(const SampledHashTable& other) = delete;

  SampledHashTable(SampledHashTable&& other)
      : _num_tables(other._num_tables),
        _reservoir_size(other._reservoir_size),
        _range(other._range),
        _max_rand(other._max_rand),
        _data(other._data),
        _counters(other._counters),
        _gen_rand(other._gen_rand) {
    other._data = nullptr;
    other._counters = nullptr;
    other._gen_rand = nullptr;
  }

  SampledHashTable& operator=(SampledHashTable&& other) {
    _num_tables = other._num_tables;
    _reservoir_size = other._reservoir_size;
    _range = other._range;
    _max_rand = other._max_rand;
    _data = other._data;
    _counters = other._counters;
    _gen_rand = other._gen_rand;

    other._data = nullptr;
    other._counters = nullptr;
    other._gen_rand = nullptr;

    return *this;
  }

  /**
   * Inserts n elements with the specified labels.
   */
  void insert(uint64_t n, const LABEL_T* labels,
              const uint32_t* hashes) override;

  /**
   * Inserts n elements with consecutive labels starting at start.
   */
  void insertSequential(uint64_t n, LABEL_T start,
                        const uint32_t* hashes) override;

  /**
   * Queries the table and returns a set that is the union of the reservoirs
   * specified by the hashes.
   */
  void queryBySet(uint32_t const* hashes,
                  std::unordered_set<LABEL_T>& store) const override;

  /**
   *
   * The hashes array should have length equal to the number of tables, and
   * the ith entry should be a bucket index into the ith table.
   *
   */

  void queryAndInsertForInference(uint32_t const* hashes,
                                  std::unordered_set<LABEL_T>& store,
                                  uint32_t outputsize);

  /**
   * Queries the table and returns the counts of elements in the union of the
   * reservoirs specified by the hashes.
   */
  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const override;

  /**
   * Queries the table and returns a vector containing the contents of the
   * reservoirs specified by the hashes.
   */
  void queryByVector(uint32_t const* hashes,
                     std::vector<LABEL_T>& results) const override;

  void clearTables() override;

  uint32_t numTables() const override { return _num_tables; };

  inline uint64_t tableRange() const override { return _range; };

  ~SampledHashTable() override;
};

}  // namespace thirdai::hashtable