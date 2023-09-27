#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include "HashTable.h"
#include <proto/hashtable.pb.h>
#include <utils/Random.h>
#include <atomic>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

class VectorHashTable final : public HashTable<uint32_t> {
 public:
  VectorHashTable(uint32_t num_tables, uint64_t table_range,
                  std::optional<uint64_t> reservoir_size = std::nullopt,
                  uint64_t seed = global_random::nextSeed(),
                  uint64_t max_rand = HashTable<uint32_t>::DEFAULT_MAX_RAND)
      : _num_tables(num_tables),
        _table_range(table_range),
        _reservoir_size(reservoir_size),
        _buckets(num_tables * table_range),
        _insertions_per_bucket(num_tables * table_range),
        _gen_rand(max_rand) {
    std::mt19937 generator(seed);
    for (uint64_t i = 0; i < max_rand; i++) {
      _gen_rand[i] = generator();
    }
  }

  explicit VectorHashTable(const proto::hashtable::VectorHashTable& hashtable);

  void insert(uint64_t n, uint32_t const* labels,
              uint32_t const* hashes) override;

  void insertSequential(uint64_t n, uint32_t start,
                        uint32_t const* hashes) override;

  void queryBySet(uint32_t const* hashes,
                  std::unordered_set<uint32_t>& store) const override;

  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const override;

  void queryByVector(uint32_t const* hashes,
                     std::vector<uint32_t>& results) const override;

  void clearTables() override;

  uint32_t numTables() const override { return _num_tables; };

  uint64_t tableRange() const override { return _table_range; };

  /** Sorts the contents of each bucket */
  void sortBuckets();

  proto::hashtable::VectorHashTable* toProto() const;

 private:
  /** Insert a label into a hashtable, including reservoir sampling if enabled
   */
  inline void insertIntoTable(uint32_t label, uint32_t hash, uint32_t table);

  uint64_t getBucketIndex(uint64_t table, uint64_t hash) const {
    return table * tableRange() + hash;
  }

  uint32_t _num_tables;
  uint64_t _table_range;
  std::optional<uint64_t> _reservoir_size;

  std::vector<std::vector<uint32_t>> _buckets;
  std::vector<uint32_t> _insertions_per_bucket;
  std::atomic_uint64_t _total_insertions = 0;

  std::vector<uint32_t> _gen_rand;

  // private constructor for cereal
  VectorHashTable() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_num_tables, _table_range, _reservoir_size, _buckets,
            _insertions_per_bucket, _gen_rand);
  }
};

}  // namespace thirdai::hashtable

CEREAL_REGISTER_TYPE(thirdai::hashtable::VectorHashTable)