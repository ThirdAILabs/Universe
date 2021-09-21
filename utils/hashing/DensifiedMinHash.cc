#include "DensifiedMinHash.h"
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
                                   uint32_t num_tables, uint32_t range_pow,
                                   uint32_t seed)
    : HashFunction(num_tables, 1 << range_pow),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _binsize(ceil(1.0 * _range / _num_hashes)) {
  _log_num_hashes = log2(_num_hashes);

  std::mt19937 gen(seed);
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  _rand_double_hash_seed = dis(gen);

  _randa = dis(gen);
  if (_randa % 2 == 0) {
    _randa++;
  }

  _random_hash = new uint32_t[2];
  _random_hash[0] = dis(gen);

  if (_random_hash[0] % 2 == 0) {
    _random_hash[0]++;
  }
  _random_hash[1] = dis(gen);
  if (_random_hash[1] % 2 == 0) {
    _random_hash[1]++;
  }

  _binids = new uint32_t[_num_hashes];

  _rand1 = new uint32_t[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes * _num_tables; i++) {
    _rand1[i] = dis(gen);
    if (_rand1[i] % 2 == 0) {
      _rand1[i]++;
    }
  }

  // int _range = 1 << this->range_power;
  // _binsize is the number of times the _range is larger than the total number
  // of hashes we need.
  for (uint32_t i = 0; i < _num_hashes; i++) {
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
  // _binsize is the number of times the _range is larger than the total number
  // of hashes we need.
  // read the data and add it to priority queue O(dlogk approx 7d) with index as
  // key and values as priority value, get TOPK index O(1) and apply minhash on
  // retuned index.

  std::priority_queue<PAIR, std::vector<PAIR>, cmp> pq;

  for (uint32_t i = 0; i < _topK; i++) {
    pq.push(std::make_pair(i, values[i]));
  }

  for (uint32_t i = _topK; i < dim; i++) {
    pq.push(std::make_pair(i, values[i]));
    pq.pop();
  }

  uint32_t* hashes = new uint32_t[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
  }

  for (uint32_t i = 0; i < _topK; i++) {
    PAIR pair = pq.top();
    pq.pop();
    uint32_t index = pair.first;
    uint32_t binid = _binids[index];
    if (hashes[binid] < index) {
      hashes[binid] = index;
    }
  }

  densifyHashes(hashes, output);
  delete[] hashes;
}

void DensifiedMinHash::hashSingleSparse(const uint32_t* indices,
                                        const float* values, uint32_t length,
                                        uint32_t* output) const {
  uint32_t* hashes = new uint32_t[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
  }

  for (uint32_t i = 0; i < length; i++) {
    uint32_t binid = _binids[indices[i]];

    if (hashes[binid] < indices[i]) {
      hashes[binid] = indices[i];
    }
  }

  densifyHashes(hashes, output);
  delete[] hashes;
  (void)values;
}

void DensifiedMinHash::densifyHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  uint32_t* hash_array = new uint32_t[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
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

  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = _rand1[_hashes_per_table * i + j];
      h *= _rand1[_hashes_per_table * i + j];
      h ^= h >> 13;
      h ^= _rand1[_hashes_per_table * i + j];
      index += h * hash_array[_hashes_per_table * i + j];
    }

    index = index & (_range - 1);
    final_hashes[i] = index;
  }
  delete[] hash_array;
}

DensifiedMinHash::~DensifiedMinHash() {
  delete[] _random_hash;
  delete[] _rand1;
  delete[] _binids;
}

}  // namespace thirdai::utils
