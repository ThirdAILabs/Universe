#include "DensifiedMinHash.h"
//#include "config.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <math.h>
#include <queue>
#include <random>
#include <stdint.h>
#include <vector>
using namespace std;

namespace bolt {

typedef pair<uint32_t, float> PAIR;

struct cmp {
  bool operator()(const PAIR& a, const PAIR& b) {
    return a.second > b.second;  // lower is better
  };
};

class SeededRandomEngine {
 private:
  static constexpr unsigned int SEED = 459386;

 public:
  SeededRandomEngine() { srand(SEED); }

  typedef unsigned int result_type;

  result_type min() { return std::numeric_limits<result_type>::min(); }

  result_type max() { return std::numeric_limits<result_type>::max(); }

  result_type operator()() { return rand(); }
};

DensifiedMinHash::DensifiedMinHash(uint32_t input_dim, uint32_t K, uint32_t L,
                                   uint32_t range_pow)
    : _K(K),
      _L(L),
      _num_hashes(K * L),
      _range(1 << range_pow),
      _binsize(ceil(1.0 * _range / _num_hashes)) {
  _log_num_hashes = log2(_num_hashes);

#ifndef SEEDED_HASHING
  std::random_device rd;
#else
  SeededRandomEngine rd;
#endif

  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  _rand_double_hash_seed = dis(gen);

  _randa = dis(gen);
  if (_randa % 2 == 0) _randa++;

  _random_hash = new uint32_t[2];
  _random_hash[0] = dis(gen);

  if (_random_hash[0] % 2 == 0) _random_hash[0]++;
  _random_hash[1] = dis(gen);
  if (_random_hash[1] % 2 == 0) _random_hash[1]++;

  _binids = new uint32_t[_num_hashes];

  _rand1 = new uint32_t[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes * _L; i++) {
    _rand1[i] = dis(gen);
    if (_rand1[i] % 2 == 0) _rand1[i]++;
  }

  // int _range = 1 << this->range_power;
  // _binsize is the number of times the _range is larger than the total number
  // of hashes we need.
  for (uint32_t i = 0; i < _num_hashes; i++) {
    uint32_t h = i;
    h *= _randa;
    h ^= h >> 13;
    h *= 0x85ebca6b;

    uint32_t curhash =
        MurmurHash((char*)&i, (uint32_t)sizeof(i), (uint32_t)_randa);
    curhash = curhash & (_range - 1);
    _binids[i] = (uint32_t)floor(curhash / _binsize);
    ;
  }

  (void)input_dim;
}

uint32_t* DensifiedMinHash::HashVector(const float* data, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[_L];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void DensifiedMinHash::HashVector(const float* data, uint32_t len,
                                  uint32_t* final_hashes) {
  // _binsize is the number of times the _range is larger than the total number
  // of hashes we need.
  // read the data and add it to priority queue O(dlogk approx 7d) with index as
  // key and values as priority value, get TOPK index O(1) and apply minhash on
  // retuned index.

  priority_queue<PAIR, vector<PAIR>, cmp> pq;

  for (uint32_t i = 0; i < _topK; i++) {
    pq.push(std::make_pair(i, data[i]));
  }

  for (uint32_t i = _topK; i < len; i++) {
    pq.push(std::make_pair(i, data[i]));
    pq.pop();
  }

  uint32_t hashes[_num_hashes];

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

  DensifyHashes(hashes, final_hashes);
  // return final_hashes;
}

uint32_t* DensifiedMinHash::HashSparseVector(const uint32_t* indices,
                                             const float* values,
                                             uint32_t len) {
  uint32_t* final_hashes = new uint32_t[_L];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void DensifiedMinHash::HashSparseVector(const uint32_t* indices,
                                        const float* values, uint32_t len,
                                        uint32_t* final_hashes) {
  uint32_t hashes[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
  }

  for (uint32_t i = 0; i < len; i++) {
    uint32_t binid = _binids[indices[i]];

    if (hashes[binid] < indices[i]) {
      hashes[binid] = indices[i];
    }
  }

  DensifyHashes(hashes, final_hashes);

  (void)values;
}

void DensifiedMinHash::DensifyHashes(uint32_t* hashes, uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t hash_array[_num_hashes] = {0};

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
      if (count > 100)  // Densification failure.
        break;
    }
    hash_array[i] = next;
  }

  for (uint32_t i = 0; i < _L; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _K; j++) {
      uint32_t h = _rand1[_K * i + j];
      h *= _rand1[_K * i + j];
      h ^= h >> 13;
      h ^= _rand1[_K * i + j];
      index += h * hash_array[_K * i + j];
    }

    index = index & (_range - 1);
    final_hashes[i] = index;
  }
}

DensifiedMinHash::~DensifiedMinHash() {
  delete[] _random_hash;
  delete[] _rand1;
  delete[] _binids;
}

}  // namespace bolt
