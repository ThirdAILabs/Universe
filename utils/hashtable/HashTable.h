#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

/**
 * This is the abstract HashTable interface. It abstracts batch indexing and
 * single queries. Implementations are intended to parallilize batch indexing.
 */
template <typename Label_t>
class HashTable {
 public:
  virtual void insert(uint64_t n, Label_t const* labels,
                      uint32_t const* hashes) = 0;

  virtual void insertSequential(uint64_t n, Label_t start,
                                uint32_t const* hashes) = 0;

  virtual void queryBySet(uint32_t const* hashes,
                          const std::unordered_set<Label_t>& store) = 0;

  virtual void queryByCount(uint32_t const* hashes,
                            const std::vector<Label_t>& counts) = 0;

  virtual void queryByVector(uint32_t const* hashes,
                             const std::vector<Label_t>& results) = 0;

  virtual void clearTables() = 0;

  virtual uint64_t numTables() = 0;

  virtual uint64_t tableRange() = 0;
};

}  // namespace thirdai::utils