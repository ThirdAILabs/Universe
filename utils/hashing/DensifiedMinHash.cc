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

// Represents a bin in which no hash was placed into during the initial step
constexpr uint32_t UNSET_HASH = UINT32_MAX;
// If this path length is reached in densification we just keep the hash unset
constexpr uint32_t MAX_DENSIFICATION_PATH_LENGTH = 100;

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
  std::vector<uint32_t> hashed_indices(length);
  for (uint32_t i = 0; i < length; i++) {
    hashed_indices[i] = MurmurHash(reinterpret_cast<const char*>(indices + i),
                                   sizeof(uint32_t), _seed);
  }
  std::sort(hashed_indices.begin(), hashed_indices.end());

  std::vector<uint32_t> hashes(_total_num_hashes, UNSET_HASH);
  uint32_t current_hash_index = 0;

  for (uint32_t bin_num = 0; bin_num < _total_num_hashes; bin_num++) {
    uint32_t bin_upper_bound = (bin_num == _total_num_hashes - 1)
                                   ? UNSET_HASH
                                   : (_binsize + 1) * bin_num;

    // Check if we have a lowest element in the bin
    if (current_hash_index < hashes.size() &&
        hashed_indices[current_hash_index] < bin_upper_bound) {
      hashes[bin_num] = hashed_indices[current_hash_index];
    }

    // Keep going until we escape the bin
    while (current_hash_index < hashes.size() &&
           hashed_indices[current_hash_index] < bin_upper_bound) {
      current_hash_index++;
    }
  }

  densifyHashes(hashes.data(), output);
}

void DensifiedMinHash::densifyHashes(uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  for (uint32_t i = 0; i < _total_num_hashes; i++) {
    uint32_t next = hashes[i];
    uint32_t count = 0;
    while (next == UNSET_HASH) {
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
