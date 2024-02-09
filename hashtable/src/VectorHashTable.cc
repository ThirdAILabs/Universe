#include "VectorHashTable.h"
#include <_types/_uint32_t.h>
#include <_types/_uint64_t.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <algorithm>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

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
      _buckets[getBucketIndex(table, hash)].size() == _reservoir_size) {
    uint32_t rand_counter = _total_insertions++;
    uint32_t num_elems_tried_insert =
        _insertions_per_bucket[getBucketIndex(table, hash)]++;
    uint32_t rand_num = _gen_rand[rand_counter % _gen_rand.size()] %
                        (num_elems_tried_insert + 1);
    if (rand_num < _reservoir_size) {
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

ar::ConstArchivePtr VectorHashTable::toArchive() const {
  auto map = ar::Map::make();

  map->set("num_tables", ar::u64(_num_tables));
  map->set("table_range", ar::u64(_table_range));

  if (_reservoir_size) {
    map->set("reservoir_size", ar::u64(*_reservoir_size));
  }

  std::vector<uint32_t> all_buckets;
  std::vector<uint64_t> bucket_offsets;
  bucket_offsets.reserve(_buckets.size() + 1);
  bucket_offsets.push_back(0);
  for (const auto& bucket : _buckets) {
    all_buckets.insert(all_buckets.end(), bucket.begin(), bucket.end());
    bucket_offsets.push_back(all_buckets.size());
  }
  map->set("buckets", ar::vecU32(std::move(all_buckets)));
  map->set("bucket_offsets", ar::vecU64(std::move(bucket_offsets)));

  map->set("insertions_per_bucket", ar::vecU32(_insertions_per_bucket));
  map->set("total_insertions", ar::u64(_total_insertions.load()));

  map->set("gen_rand", ar::vecU32(_gen_rand));

  return map;
}

std::shared_ptr<VectorHashTable> VectorHashTable::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<VectorHashTable>(archive);
}

VectorHashTable::VectorHashTable(const ar::Archive& archive)
    : _num_tables(archive.u64("num_tables")),
      _table_range(archive.u64("table_range")),
      _reservoir_size(archive.getOpt<ar::U64>("reservoir_size")),
      _buckets(_num_tables * _table_range),
      _insertions_per_bucket(
          archive.getAs<ar::VecU32>("insertions_per_bucket")),
      _total_insertions(archive.getAs<ar::U64>("total_insertions")),
      _gen_rand(archive.getAs<ar::VecU32>("gen_rand")) {
  const auto& all_buckets = archive.getAs<ar::VecU32>("buckets");
  const auto& bucket_offsets = archive.getAs<ar::VecU64>("bucket_offsets");

  for (size_t i = 0; i < _buckets.size(); i++) {
    _buckets[i] = {all_buckets.begin() + bucket_offsets[i],
                   all_buckets.begin() + bucket_offsets[i + 1]};
  }
}

}  // namespace thirdai::hashtable
