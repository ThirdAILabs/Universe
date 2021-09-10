#include "NewSRP.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

namespace bolt {

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

constexpr uint32_t DEFAULT_BINSIZE = 8;

FastSRP::FastSRP(uint32_t input_dim, uint32_t hashes_per_table,
                 uint32_t num_tables, uint32_t range_pow)
    : _hashes_per_table(hashes_per_table),
      _num_tables(num_tables),
      _num_hashes(hashes_per_table * num_tables),
      _dim(input_dim),
      _binsize(DEFAULT_BINSIZE) {
  (void)range_pow;

  _permute = ceil(((double)_num_hashes * _binsize) / _dim);
  _log_num_hashes = log2(_num_hashes);

#ifndef SEEDED_HASHING
  std::random_device rd;
#else
  SeededRandomEngine rd;
#endif

  std::mt19937 gen(rd());
  uint32_t* n_array = new uint32_t[_dim];
  _bin_map = new uint32_t[_dim * _permute];
  _positions = new uint32_t[_dim * _permute];
  _rand_bits = new short[_dim * _permute];

  for (uint32_t i = 0; i < _dim; i++) {
    n_array[i] = i;
  }

  for (uint32_t i = 0; i < _dim * _permute; i++) {
    uint32_t curr = rand();
    if (curr % 2 == 0) {
      _rand_bits[i] = 1;
    } else {
      _rand_bits[i] = -1;
    }
  }

  for (uint32_t p = 0; p < _permute; p++) {
    std::shuffle(n_array, n_array + _dim, rd);
    for (uint32_t j = 0; j < _dim; j++) {
      _bin_map[p * _dim + n_array[j]] = (p * _dim + j) / _binsize;
      _positions[p * _dim + n_array[j]] = (p * _dim + j) % _binsize;
    }
  }

  delete[] n_array;

  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  _rand_double_hash_seed = dis(gen);
}

uint32_t* FastSRP::HashVector(const float* data, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[_num_tables];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void FastSRP::HashVector(const float* data, uint32_t len,
                         uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t hashes[_num_hashes];
  float bin_values[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = _bin_map[base_bin_id + i];
      // if (binid < _num_hashes && bin_values[binid] < data[i]) {
      //   bin_values[binid] = data[i];
      //   hashes[binid] = _positions[base_bin_id + i];
      // }
      if (binid < _num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = data[i] * _rand_bits[binid];
        } else {
          bin_values[binid] += data[i] * _rand_bits[binid];
        }
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }
    }
  }

  DensifyHashes(hashes, final_hashes);
}

uint32_t* FastSRP::HashSparseVector(const uint32_t* indices,
                                    const float* values, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[_num_tables];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void FastSRP::HashSparseVector(const uint32_t* indices, const float* values,
                               uint32_t len, uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t hashes[_num_hashes];
  float bin_values[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = _bin_map[base_bin_id + indices[i]];
      // if (binid < _num_hashes && bin_values[binid] < values[i]) {
      //   bin_values[binid] = values[i];
      //   hashes[binid] = _positions[base_bin_id + indices[i]];
      // }
      if (binid < _num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = values[i] * _rand_bits[binid];
        } else {
          bin_values[binid] += values[i] * _rand_bits[binid];
        }
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }
    }
  }

  DensifyHashes(hashes, final_hashes);
}

void FastSRP::DensifyHashes(uint32_t* hashes, uint32_t* final_hashes) {
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

  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = hash_array[i * _hashes_per_table + j];
      index += h << (_hashes_per_table - 1 - j);
    }
    final_hashes[i] = index;
  }
}

FastSRP::~FastSRP() {
  delete[] _bin_map;
  delete[] _positions;
}

}  // namespace bolt