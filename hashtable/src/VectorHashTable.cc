#include "VectorHashTable.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

template <typename LABEL_T, bool USE_RESERVOIR>
VectorHashTable<LABEL_T, USE_RESERVOIR>::VectorHashTable(
    const proto::hashtable::VectorHashTable& hashtable)
    : _num_tables(hashtable.num_tables()),
      _table_range(hashtable.table_range()),
      _generated_rand_nums(hashtable.gen_rand().begin(),
                           hashtable.gen_rand().end()),
      _num_elements_tried_insert_into_bucket(
          hashtable.insertions_per_bucket().begin(),
          hashtable.insertions_per_bucket().end()),
      counter(hashtable.total_insertions()) {
  if (hashtable.has_reservoir_size()) {
    _max_reservoir_size = hashtable.reservoir_size();
  } else {
    _max_reservoir_size = 0;
  }

  _buckets.reserve(hashtable.buckets_size());
  for (const auto& bucket : hashtable.buckets()) {
    _buckets.emplace_back(bucket.ids().begin(), bucket.ids().end());
  }
}

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

template <typename LABEL_T, bool USE_RESERVOIR>
proto::hashtable::VectorHashTable*
VectorHashTable<LABEL_T, USE_RESERVOIR>::toProto() const {
  auto* hashtable = new proto::hashtable::VectorHashTable();

  hashtable->set_num_tables(_num_tables);
  hashtable->set_table_range(_table_range);
  if (USE_RESERVOIR) {
    hashtable->set_reservoir_size(_max_reservoir_size);
  }

  for (const auto& bucket : _buckets) {
    auto* bucket_proto = hashtable->add_buckets();
    *bucket_proto->mutable_ids() = {bucket.begin(), bucket.end()};
  }

  *hashtable->mutable_gen_rand() = {_generated_rand_nums.begin(),
                                    _generated_rand_nums.end()};

  *hashtable->mutable_insertions_per_bucket() = {
      _num_elements_tried_insert_into_bucket.begin(),
      _num_elements_tried_insert_into_bucket.end()};

  hashtable->set_total_insertions(counter);

  return hashtable;
}

template class VectorHashTable<uint8_t, true>;
template class VectorHashTable<uint16_t, true>;
template class VectorHashTable<uint32_t, true>;
template class VectorHashTable<uint64_t, true>;
template class VectorHashTable<uint8_t, false>;
template class VectorHashTable<uint16_t, false>;
template class VectorHashTable<uint32_t, false>;
template class VectorHashTable<uint64_t, false>;

}  // namespace thirdai::hashtable
