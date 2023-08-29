#include "VectorHashTable.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

VectorHashTable::VectorHashTable(
    const proto::hashtable::VectorHashTable& hashtable)
    : _num_tables(hashtable.num_tables()),
      _table_range(hashtable.table_range()),
      _insertions_per_bucket(hashtable.insertions_per_bucket().begin(),
                             hashtable.insertions_per_bucket().end()),
      _total_insertions(hashtable.total_insertions()),
      _gen_rand(hashtable.gen_rand().begin(), hashtable.gen_rand().end()) {
  if (hashtable.has_reservoir_size()) {
    _reservoir_size = hashtable.reservoir_size();
  }

  _buckets.reserve(hashtable.buckets_size());
  for (const auto& bucket : hashtable.buckets()) {
    _buckets.emplace_back(bucket.ids().begin(), bucket.ids().end());
  }
}

void VectorHashTable::insert(uint64_t n, uint32_t const* labels,
                             uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, hashes, labels)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      uint32_t label = labels[item];
      insertIntoTable(label, hash, table);
    }
  }
}

void VectorHashTable::insertSequential(uint64_t n, uint32_t start,
                                       uint32_t const* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint64_t item = 0; item < n; item++) {
      uint32_t hash = hashes[_num_tables * item + table];
      uint32_t label = start + item;
      insertIntoTable(label, hash, table);
    }
  }
}

inline void VectorHashTable::insertIntoTable(uint32_t label, uint32_t hash,
                                             uint32_t table) {
  if (_reservoir_size &&
      _buckets[getBucketIndex(table, hash)].size() == *_reservoir_size) {
    uint32_t rand_counter = _total_insertions++;
    uint32_t num_elems_tried_insert =
        _insertions_per_bucket[getBucketIndex(table, hash)]++;
    uint32_t rand_num = _gen_rand[rand_counter % *_reservoir_size] %
                        (num_elems_tried_insert + 1);
    if (rand_num < *_reservoir_size) {
      _buckets[getBucketIndex(table, hash)][rand_num] = label;
    }
  } else {
    _buckets[getBucketIndex(table, hash)].push_back(label);
  }
}

void VectorHashTable::queryBySet(uint32_t const* hashes,
                                 std::unordered_set<uint32_t>& store) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (uint32_t label : _buckets[getBucketIndex(table, hash)]) {
      store.insert(label);
    }
  }
}

void VectorHashTable::queryByCount(uint32_t const* hashes,
                                   std::vector<uint32_t>& counts) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (uint32_t label : _buckets[getBucketIndex(table, hash)]) {
      counts[label]++;
    }
  }
}

void VectorHashTable::queryByVector(uint32_t const* hashes,
                                    std::vector<uint32_t>& results) const {
  for (uint32_t table = 0; table < _num_tables; table++) {
    uint32_t hash = hashes[table];
    for (uint32_t label : _buckets[getBucketIndex(table, hash)]) {
      results.push_back(label);
    }
  }
}

void VectorHashTable::clearTables() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    _buckets[index].clear();
  }
}

void VectorHashTable::sortBuckets() {
  for (uint64_t index = 0; index < _num_tables * _table_range; index++) {
    std::sort(_buckets[index].begin(), _buckets[index].end());
  }
}

proto::hashtable::VectorHashTable* VectorHashTable::toProto() const {
  auto* hashtable = new proto::hashtable::VectorHashTable();

  hashtable->set_num_tables(_num_tables);
  hashtable->set_table_range(_table_range);
  if (_reservoir_size) {
    hashtable->set_reservoir_size(*_reservoir_size);
  }

  for (const auto& bucket : _buckets) {
    auto* bucket_proto = hashtable->add_buckets();
    *bucket_proto->mutable_ids() = {bucket.begin(), bucket.end()};
  }

  *hashtable->mutable_gen_rand() = {_gen_rand.begin(), _gen_rand.end()};

  *hashtable->mutable_insertions_per_bucket() = {_insertions_per_bucket.begin(),
                                                 _insertions_per_bucket.end()};

  hashtable->set_total_insertions(_total_insertions.load());

  return hashtable;
}

}  // namespace thirdai::hashtable
