#pragma once

namespace thirdai::hashtable {

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::insert(uint64_t n,
                                                     LABEL_T const* labels,
                                                     uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, hashes, labels)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      LABEL_T label = labels[item];
      insertIntoTable(label, hash, table);
    }
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::insertSequential(
    uint64_t n, LABEL_T start, uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      LABEL_T label = start + item;
      insertIntoTable(label, hash, table);
    }
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
inline void VectorHashTable<LABEL_T, USE_RESERVOIR>::insertIntoTable(
    LABEL_T label, uint32_t hash, uint32_t table) {
  if (USE_RESERVOIR &&
      _buckets[getBucketIndex(table, hash)].size() == _max_reservoir_size) {
    uint32_t rand_counter = counter++;
    uint32_t num_elems_tried_insert =
        _num_elements_tried_insert_into_bucket[getBucketIndex(table, hash)]++;
    uint32_t rand_num =
        _generated_rand_nums[rand_counter % _max_reservoir_size] %
        (num_elems_tried_insert + 1);
    if (rand_num < _max_reservoir_size) {
      _buckets[getBucketIndex(table, hash)][rand_num] = label;
    }
  } else {
    _buckets[getBucketIndex(table, hash)].push_back(label);
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::queryBySet(
    uint32_t const* hashes, std::unordered_set<LABEL_T>& store) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (LABEL_T label : _buckets[getBucketIndex(table, hash)]) {
      store.insert(label);
    }
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::queryByCount(
    uint32_t const* hashes, std::vector<uint32_t>& counts) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (LABEL_T label : _buckets[getBucketIndex(table, hash)]) {
      counts[label]++;
    }
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::queryByVector(
    uint32_t const* hashes, std::vector<LABEL_T>& results) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (LABEL_T label : _buckets[getBucketIndex(table, hash)]) {
      results.push_back(label);
    }
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::clearTables() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    _buckets[index].clear();
  }
}

template <typename LABEL_T, bool USE_RESERVOIR>
void VectorHashTable<LABEL_T, USE_RESERVOIR>::sortBuckets() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    std::sort(_buckets[index].begin(), _buckets[index].end());
  }
}

}  // namespace thirdai::hashtable
