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

FastSRP::FastSRP(uint32_t input_dim, uint32_t _K, uint32_t _L, uint32_t range_pow)
    : K(_K),
      L(_L),
      num_hashes(K * L),
      range(1 << K),
      dim(input_dim),
      binsize(DEFAULT_BINSIZE) {
  (void) range_pow;

  permute = ceil(((double)num_hashes * binsize) / dim);
  log_num_hashes = log2(num_hashes);
  log_binsize = floor(log2(binsize));

#ifndef SEEDED_HASHING
  std::random_device rd;
#else
  SeededRandomEngine rd;
#endif

  std::mt19937 gen(rd());
  uint32_t* n_array = new uint32_t[dim];
  bin_map = new uint32_t[dim * permute];
  positions = new uint32_t[dim * permute];
  rand_bits = new short[dim * permute];

  for (uint32_t i = 0; i < dim; i++) {
    n_array[i] = i;
  }

  for (uint32_t i = 0; i < dim * permute; i++) {
    uint32_t curr = rand();
    if (curr % 2 == 0) {
      rand_bits[i] = 1;
    } else {
      rand_bits[i] = -1;
    }
  }

  for (uint32_t p = 0; p < permute; p++) {
    std::shuffle(n_array, n_array + dim, rd);
    for (uint32_t j = 0; j < dim; j++) {
      bin_map[p * dim + n_array[j]] = (p * dim + j) / binsize;
      positions[p * dim + n_array[j]] = (p * dim + j) % binsize;
    }
  }

  delete[] n_array;

  std::uniform_int_distribution<uint32_t> dis(1, std::numeric_limits<uint32_t>::max() - 1);

  rand_double_hash_seed = dis(gen);
}

uint32_t* FastSRP::HashVector(const float* data, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void FastSRP::HashVector(const float* data, uint32_t len, uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory allocation
  uint32_t hashes[num_hashes];
  float bin_values[num_hashes];

  for (uint32_t i = 0; i < num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < permute; p++) {
    uint32_t base_bin_id = p * dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = bin_map[base_bin_id + i];
      // if (binid < num_hashes && bin_values[binid] < data[i]) {
      //   bin_values[binid] = data[i];
      //   hashes[binid] = positions[base_bin_id + i];
      // }
      if (binid < num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = data[i] * rand_bits[binid];
        } else {
          bin_values[binid] += data[i] * rand_bits[binid];
        }  
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }

    }
  }

  DensifyHashes(hashes, final_hashes);
}

uint32_t* FastSRP::HashSparseVector(const uint32_t* indices, const float* values,
                                             uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void FastSRP::HashSparseVector(const uint32_t* indices, const float* values, uint32_t len,
                                        uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory allocation
  uint32_t hashes[num_hashes];
  float bin_values[num_hashes];

  for (uint32_t i = 0; i < num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < permute; p++) {
    uint32_t base_bin_id = p * dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = bin_map[base_bin_id + indices[i]];
      // if (binid < num_hashes && bin_values[binid] < values[i]) {
      //   bin_values[binid] = values[i];
      //   hashes[binid] = positions[base_bin_id + indices[i]];
      // }
      if (binid < num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = values[i] * rand_bits[binid];
        } else {
          bin_values[binid] += values[i] * rand_bits[binid];
        }  
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }
    }
  }

  DensifyHashes(hashes, final_hashes);
}

void FastSRP::DensifyHashes(uint32_t* hashes, uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory allocation
  uint32_t hash_array[num_hashes] = {0};

  for (uint32_t i = 0; i < num_hashes; i++) {
    uint32_t next = hashes[i];
    if (next != std::numeric_limits<uint32_t>::max()) {
      hash_array[i] = hashes[i];
      continue;
    }

    uint32_t count = 0;
    while (next == std::numeric_limits<uint32_t>::max()) {
      count++;
      uint32_t index = std::min(RandDoubleHash(i, count), num_hashes);

      next = hashes[index];
      if (count > 100)  // Densification failure.
        break;
    }
    hash_array[i] = next;
  }

  for (uint32_t i = 0; i < L; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < K; j++) {
      uint32_t h = hash_array[i * K + j];
      index += h << (K - 1 - j);
    }
    final_hashes[i] = index;
  }
}

FastSRP::~FastSRP() {
  delete[] bin_map;
  delete[] positions;
}

}  // namespace bolt