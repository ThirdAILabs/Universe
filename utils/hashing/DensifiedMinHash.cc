#include "DensifiedMinHash.h"
#include "../Exceptions.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

namespace thirdai::utils {

// If this path length is reached in densification we just keep the hash unset
constexpr uint32_t MAX_DENSIFICATION_PATH_LENGTH = 100;

/**
 * TODO(josh): __builtin_ffs is not an obvious thing (it returns the index of
 * first bit set + 1). We should add a better named function to a util file.
 */
DensifiedMinHash::DensifiedMinHash(uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t seed)
    : HashFunction(num_tables, UINT32_MAX),
      _hashes_per_table(hashes_per_table),
      _total_num_hashes(hashes_per_table * num_tables),
      _binsize(_range / _total_num_hashes),
      _seed(seed),
      _log_2_num_hashes(__builtin_ffs(_total_num_hashes) - 1) {
  if (1U << _log_2_num_hashes != _total_num_hashes) {
    throw std::invalid_argument(
        "The total number of hashes (hashes_per_table * num_tables) must be a "
        "power of 2, but was " +
        std::to_string(_total_num_hashes));
  }
}

void DensifiedMinHash::hashSingleDense(const float* values, uint32_t dim,
                                       uint32_t* output) const {
  (void)values;
  (void)dim;
  (void)output;
  throw thirdai::utils::NotImplemented();
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
  // consecutive hash values. Another complicating factor is that we would need
  // to know themax dimension upon construction to initialize the range and
  // bin size.
  std::vector<uint32_t> hashes(_total_num_hashes, UINT32_MAX);
  for (uint32_t i = 0; i < length; i++) {
    uint32_t hash = MurmurHash(reinterpret_cast<const char*>(indices + i),
                                    sizeof(uint32_t), _seed);
    uint32_t bin_id = std::min(hash / _binsize, _total_num_hashes - 1);
    hashes[bin_id] = std::min(hash, hashes[bin_id]);
  }

  densifyHashes(hashes.data(), output);
}

void DensifiedMinHash::densifyHashes(uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  for (uint32_t i = 0; i < _total_num_hashes; i++) {
    uint32_t next = hashes[i];
    uint32_t count = 0;
    while (next == UINT32_MAX) {
      count++;
      uint32_t index = fastDoubleHash(i, count, _log_2_num_hashes);
      next = hashes[index];
      if (count > MAX_DENSIFICATION_PATH_LENGTH) {  // Densification failure.
        break;
      }
    }
    hashes[i] = next;
  }

  compactHashes(hashes, final_hashes);
}

void DensifiedMinHash::compactHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = hashes[i * _hashes_per_table + j];
      index += h << (_hashes_per_table - 1 - j);
    }
    final_hashes[i] = index;
  }
}

}  // namespace thirdai::utils
