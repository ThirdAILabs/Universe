#include "MinHash.h"
#include "HashUtils.h"
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <exceptions/src/Exceptions.h>
#include <limits>
#include <random>

namespace thirdai::hashing {

MinHash::MinHash(uint32_t hashes_per_table, uint32_t num_tables, uint32_t range,
                 uint32_t seed)
    : HashFunction(num_tables, range),
      _hashes_per_table(hashes_per_table),
      _total_num_hashes(hashes_per_table * num_tables) {
  std::mt19937 rng(seed);

  for (uint32_t i = 0; i < _total_num_hashes; i++) {
    _hash_functions.push_back(UniversalHash(rng()));
  }
}

void MinHash::hashSingleSparse(const uint32_t* indices, const float* values,
                               uint32_t length, uint32_t* output) const {
  (void)values;

  std::vector<uint32_t> all_hashes(_total_num_hashes);

  for (uint32_t hash_idx = 0; hash_idx < _total_num_hashes; hash_idx++) {
    uint32_t min_hash = std::numeric_limits<uint32_t>::max();

    for (uint32_t i = 0; i < length; i++) {
      uint32_t hash = _hash_functions[hash_idx].gethash(indices[i]);
      min_hash = std::min(min_hash, hash);
    }

    all_hashes[hash_idx] = min_hash;
  }

  defaultCompactHashes(all_hashes.data(), output, _num_tables,
                       _hashes_per_table);

  for (uint32_t t = 0; t < _num_tables; t++) {
    output[t] %= _range;
  }
}

void MinHash::hashSingleDense(const float* values, uint32_t dim,
                              uint32_t* output) const {
  (void)values;
  (void)dim;
  (void)output;
  throw thirdai::exceptions::NotImplemented(
      "DensifiedMinHash cannot hash dense arrays.");
}

ar::ConstArchivePtr MinHash::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("num_tables", ar::u64(_num_tables));
  map->set("hashes_per_table", ar::u64(_hashes_per_table));
  map->set("range", ar::u64(_range));

  auto hash_fns = ar::List::make();
  for (const auto& hash_fn : _hash_functions) {
    hash_fns->append(hash_fn.toArchive());
  }
  map->set("hash_functions", hash_fns);

  return map;
}

std::shared_ptr<MinHash> MinHash::fromArchive(const ar::Archive& archive) {
  return std::make_shared<MinHash>(archive);
}

MinHash::MinHash(const ar::Archive& archive)
    : HashFunction(archive.u64("num_tables"), archive.u64("range")),
      _hashes_per_table(archive.u64("hashes_per_table")) {
  _total_num_hashes = _num_tables * _hashes_per_table;

  for (const auto& hash_fn : archive.get("hash_functions")->list()) {
    _hash_functions.emplace_back(*hash_fn);
  }
}

}  // namespace thirdai::hashing