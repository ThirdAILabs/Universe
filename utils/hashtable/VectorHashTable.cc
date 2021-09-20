#include "VectorHashTable.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

template class VectorHashTable<uint8_t>;
template class VectorHashTable<uint16_t>;
template class VectorHashTable<uint32_t>;
template class VectorHashTable<uint64_t>;

template <typename Label_t>
VectorHashTable<Label_t>::VectorHashTable(uint32_t num_tables,
                                          uint64_t table_range)
    : _num_tables(num_tables),
      _table_range(table_range),
      tables(num_tables * table_range) {}

template <typename Label_t>
void VectorHashTable<Label_t>::insert(uint64_t n, Label_t const* labels,
                                      uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, hashes, labels)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      Label_t label = labels[item];
      tables[getBucketIndex(table, hash)].push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::insertSequential(uint64_t n, Label_t start,
                                                uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      Label_t label = start + item;
      tables[getBucketIndex(table, hash)].push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::queryBySet(
    uint32_t const* hashes, std::unordered_set<Label_t>& store) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (Label_t label : tables[getBucketIndex(table, hash)]) {
      store.insert(label);
    }
  }
}

template <typename Label_t>
void VectorHashTable<Label_t>::queryByCount(
    uint32_t const* hashes, std::vector<uint32_t>& counts) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (Label_t label : tables[getBucketIndex(table, hash)]) {
      counts[label]++;
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::queryByVector(
    uint32_t const* hashes, std::vector<Label_t>& results) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (Label_t label : tables[getBucketIndex(table, hash)]) {
      results.push_back(label);
    }
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::clearTables() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    tables[index].clear();
  }
};

template <typename Label_t>
void VectorHashTable<Label_t>::sortBuckets() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    std::sort(tables[index].begin(), tables[index].end());
  }
}

template <typename Label_t>
uint32_t VectorHashTable<Label_t>::numTables() const {
  return _num_tables;
}

template <typename Label_t>
uint64_t VectorHashTable<Label_t>::tableRange() const {
  return _table_range;
}

}  // namespace thirdai::utils