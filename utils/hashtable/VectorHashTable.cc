#include "VectorHashTable.h"
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

template <typename Label_t>
VectorHashTable<Label_t>::VectorHashTable(uint64_t num_tables,
                                          uint64_t table_range)
    : _num_tables(num_tables),
      _table_range(table_range),
      tables(num_tables * table_range) {}

template <typename Label_t>
void VectorHashTable<Label_t>::insert(uint64_t n, Label_t const* labels,
                                      uint32_t const* hashes) {
#pragma omp parallel for default(none)
  for (uint64_t table = 0; table < _num_tables; ++table) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[table * n + item];
      Label_t label = labels[item];
      getBucket(table, hash).push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::insertSequential(uint64_t n, Label_t start,
                                                uint32_t const* hashes) {
#pragma omp parallel for default(none)
  for (uint64_t table = 0; table < _num_tables; ++table) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[table * n + item];
      Label_t label = start + item;
      getBucket(table, hash).push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::queryBySet(
    uint32_t const* hashes, const std::unordered_set<Label_t>& store) {
  for (uint64_t table = 0; table < _num_tables; ++table) {
    uint32_t hash = hashes[table];
    for (Label_t label : getBucket(table, hash)) {
      store.insert(label);
    }
  }
}

template <typename Label_t>
void VectorHashTable<Label_t>::queryByCount(
    uint32_t const* hashes, const std::vector<Label_t>& counts) {
  for (uint64_t table = 0; table < _num_tables; ++table) {
    uint32_t hash = hashes[table];
    for (Label_t label : getBucket(table, hash)) {
      ++counts[label];
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::queryByVector(
    uint32_t const* hashes, const std::vector<Label_t>& results) {
  for (uint64_t table = 0; table < _num_tables; ++table) {
    uint32_t hash = hashes[table];
    for (Label_t label : getBucket(table, hash)) {
      results.push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::clearTables() {
  for (auto vector : tables) {
    vector.clear();
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::sortBuckets() {
  for (auto vector : tables) {
    sort(vector.begin(), vector.end());
  }
}

template <typename Label_t>
uint64_t VectorHashTable<Label_t>::numTables() {
  return _num_tables;
}

template <typename Label_t>
uint64_t VectorHashTable<Label_t>::tableRange() {
  return _table_range;
}

}  // namespace thirdai::utils