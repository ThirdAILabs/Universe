#include "DensifiedMinHash.h"
#include "HashUtils.h"
#include <exceptions/src/Exceptions.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::hashing {

DensifiedMinHash::DensifiedMinHash(uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t range,
                                   uint32_t seed)
    : HashFunction(num_tables, range),
      _hashes_per_table(hashes_per_table),
      _total_num_hashes(hashes_per_table * num_tables),
      _binsize(UINT32_MAX / _total_num_hashes),
      _seed(seed) {}

void DensifiedMinHash::hashSingleDense(const float* values, uint32_t dim,
                                       uint32_t* output) const {
  (void)values;
  (void)dim;
  (void)output;
  throw thirdai::exceptions::NotImplemented(
      "DensifiedMinHash cannot hash dense arrays.");
}

void DensifiedMinHash::hashSingleSparse(const uint32_t* indices,
                                        const float* values, uint32_t length,
                                        uint32_t* output) const {
  (void)values;

  // If murmur hash takes too long, one thing we can do is switch to a faster
  // hash. Another interesting alternative is to just use the original indices
  // and sort those. This is not preferred, however, because there might be
  // locality and thus information loss in the original indices, i.e. having
  // indices 899 890 891 892 is probably much more likely than having 4
  // consecutive hash values. Another complicating factor is that we would
  // need to know themax dimension upon construction to initialize the range
  // and bin size.
  std::vector<uint32_t> hashes(_total_num_hashes, UINT32_MAX);
  for (uint32_t i = 0; i < length; i++) {
    uint32_t hash = MurmurHash(reinterpret_cast<const char*>(indices + i),
                               sizeof(uint32_t), _seed);
    uint32_t bin_id = std::min(hash / _binsize, _total_num_hashes - 1);
    hashes[bin_id] = std::min(hash, hashes[bin_id]);
  }

  HashUtils::densifyHashes(hashes.data(), _total_num_hashes);
  HashUtils::defaultCompactHashes(hashes.data(), output, _num_tables,
                                  _hashes_per_table);
  for (uint32_t i = 0; i < _num_tables; i++) {
    output[i] %= _range;
  }
}

}  // namespace thirdai::hashing
