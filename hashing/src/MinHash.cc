#include "MinHash.h"
#include "HashUtils.h"
#include <hashing/src/HashFunction.h>
#include <exceptions/src/Exceptions.h>
#include <proto/hashing.pb.h>
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

MinHash::MinHash(const proto::hashing::MinHash& minhash)
    : HashFunction(minhash.num_tables(), minhash.range()),
      _hashes_per_table(minhash.hashes_per_table()),
      _total_num_hashes(minhash.num_tables() * minhash.hashes_per_table()) {
  for (const auto& hash_fn : minhash.hash_functions()) {
    _hash_functions.emplace_back(hash_fn);
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

proto::hashing::HashFunction* MinHash::toProto() const {
  auto* hash_fn = new proto::hashing::HashFunction();
  auto* minhash = hash_fn->mutable_minhash();

  minhash->set_num_tables(_num_tables);
  minhash->set_hashes_per_table(_hashes_per_table);
  minhash->set_range(_range);

  for (const auto& universal_hash : _hash_functions) {
    minhash->mutable_hash_functions()->AddAllocated(universal_hash.toProto());
  }

  return hash_fn;
}

}  // namespace thirdai::hashing