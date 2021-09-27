#include "DensifiedMinHash.h"
#include "../Exceptions.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

namespace thirdai::utils {

typedef std::pair<uint32_t, float> PAIR;

struct cmp {
  bool operator()(const PAIR& a, const PAIR& b) {
    return a.second > b.second;  // lower is better
  };
};

DensifiedMinHash::DensifiedMinHash(uint32_t input_dim,
                                   uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t seed)
    : HashFunction(num_tables, UINT32_MAX),
      _hashes_per_table(hashes_per_table),
      _total_num_hashes(hashes_per_table * num_tables),
      _binsize(_range / _total_num_hashes),
      _seed(seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  uint32_t _randa = dis(gen);
  if (_randa % 2 == 0) {
    _randa++;
  }

  _binids = new uint32_t[_total_num_hashes];

  pregenerated_random_nums = new uint32_t[pregenerated_random_nums];

  for (uint32_t i = 0; i < pregenerated_random_nums; i++) {
    pregenerated_random_nums[i] = dis(gen);
  }

  // int _range = 1 << this->range_power;
  // _binsize is the number of times the _range is larger than the total number
  // of hashes we need.
  for (uint32_t i = 0; i < pregenerated_random_nums; i++) {
    uint32_t curhash = MurmurHash(reinterpret_cast<char*>(&i),
                                  static_cast<uint32_t>(sizeof(i)),
                                  static_cast<uint32_t>(_randa));
    curhash = curhash & (_range - 1);
    _binids[i] = floor(static_cast<double>(curhash) / _binsize);
  }

  (void)input_dim;
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

  // If murmur hash takes too long, we can just use the original indices and
  // sort those. This is not preferred, however, because there might be locality
  // and thus information loss in the original indices, i.e. having indices 899
  // 890 891 892 is probably much more likely than having 4 consecutive hash
  // values. Another complicating factor is that we would need to know the
  // max dimension upon construction to initialize the range and bin size.
  std::vector<uint32_t> hashed_indices(length);
  for (uint32_t i = 0; i < length; i++) {
    hashed_indices[i] = MurmurHash(reinterpret_cast<const char*>(indices + i),
                                   sizeof(uint32_t), _seed);
  }
  std::sort(hashed_indices.begin(), hashed_indices.end());

  // Here, UINT32_MAX represents the absence of a hash
  // TODO(Josh) bring this into a util method
  std::vector<uint32_t> hashes(_total_num_hashes, UINT32_MAX);
  uint32_t current_hash_index = 0;
  for (uint32_t bin_num = 0; bin_num < _total_num_hashes; bin_num++) {
    uint32_t bin_upper_bound = bin_num == _total_num_hashes - 1
                                   ? UINT32_MAX
                                   : (_binsize + 1) * bin_num;

    // Check if we have a lowest element in the bin
    if (current_hash_index < hashes.size() &&
        hashed_indices[current_hash_index] < bin_upper_bound) {
      hashes[bin_num] = hashed_indices[current_hash_index];
    }

    while (current_hash_index < hashes.size() &&
           hashed_indices[current_hash_index] < bin_upper_bound) {
      current_hash_index++;
    }
  }

  densifyHashes(hashes.data(), output);
}

void DensifiedMinHash::densifyHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  uint32_t* hash_array = new uint32_t[_total_num_hashes]();

  for (uint32_t i = 0; i < _total_num_hashes; i++) {
    uint32_t next = hashes[i];
    if (next != std::numeric_limits<uint32_t>::max()) {
      hash_array[i] = hashes[i];
      continue;
    }

    uint32_t count = 0;
    while (next == std::numeric_limits<uint32_t>::max()) {
      count++;
      uint32_t index = std::min(RandDoubleHash(i, count), _num_hashes);

      next = hashes[index];
      if (count > 100) {  // Densification failure.
        break;
      }
    }
    hash_array[i] = next;
  }

  compactHashes(hash_array, final_hashes);

  delete[] hash_array;
}

void DensifiedMinHash::compactHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = _rand1[_hashes_per_table * i + j];
      h *= _rand1[_hashes_per_table * i + j];
      h ^= h >> 13;
      h ^= _rand1[_hashes_per_table * i + j];
      index += h * hashes[_hashes_per_table * i + j];
    }

    index = index & (_range - 1);
    final_hashes[i] = index;
  }
}

DensifiedMinHash::~DensifiedMinHash() {
  delete[] _rand1;
  delete[] _binids;
}

}  // namespace thirdai::utils
