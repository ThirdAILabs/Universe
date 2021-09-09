#include "DWTA.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

namespace thirdai::bolt {

constexpr uint32_t DEFAULT_BINSIZE = 8;

DWTAHashFunction::DWTAHashFunction(uint32_t input_dim, uint32_t _K, uint32_t _L,
                                   uint32_t range_pow)
    : K(_K),
      L(_L),
      num_hashes(K * L),
      range(1 << range_pow),
      dim(input_dim),
      binsize(DEFAULT_BINSIZE) {
  this->permute = ceil(((double)num_hashes * binsize) / dim);
  this->log_num_hashes = log2(num_hashes);
  this->log_binsize = floor(log2(binsize));

#ifndef SEEDED_HASHING
  std::random_device rd;
#else
  SeededRandomEngine rd;
#endif

  std::mt19937 gen(rd());
  uint32_t* n_array = new uint32_t[dim];
  bin_map = new uint32_t[dim * permute];
  positions = new uint32_t[dim * permute];

  for (uint32_t i = 0; i < dim; i++) {
    n_array[i] = i;
  }
  for (uint32_t p = 0; p < permute; p++) {
    std::shuffle(n_array, n_array + dim, rd);
    for (uint32_t j = 0; j < dim; j++) {
      bin_map[p * dim + n_array[j]] = (p * dim + j) / binsize;
      positions[p * dim + n_array[j]] = (p * dim + j) % binsize;
    }
  }

  delete[] n_array;

  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  rand_double_hash_seed = dis(gen);
}

uint32_t* DWTAHashFunction::HashVector(const float* data, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void DWTAHashFunction::HashVector(const float* data, uint32_t len,
                                  uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t* hashes = new uint32_t[num_hashes];
  float* bin_values = new float[num_hashes];

  for (uint32_t i = 0; i < num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < permute; p++) {
    uint32_t base_bin_id = p * dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = bin_map[base_bin_id + i];
      if (binid < num_hashes && bin_values[binid] < data[i]) {
        bin_values[binid] = data[i];
        hashes[binid] = positions[base_bin_id + i];
      }
    }
  }

  delete[] bin_values;

  DensifyHashes(hashes, final_hashes);

  delete[] hashes;
}

uint32_t* DWTAHashFunction::HashSparseVector(const uint32_t* indices,
                                             const float* values,
                                             uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void DWTAHashFunction::HashSparseVector(const uint32_t* indices,
                                        const float* values, uint32_t len,
                                        uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t* hashes = new uint32_t[num_hashes];
  float* bin_values = new float[num_hashes];

  for (uint32_t i = 0; i < this->num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < this->permute; p++) {
    uint32_t base_bin_id = p * dim;
    for (uint32_t i = 0; i < len; i++) {
      uint32_t binid = this->bin_map[base_bin_id + indices[i]];
      if (binid < this->num_hashes && bin_values[binid] < values[i]) {
        bin_values[binid] = values[i];
        hashes[binid] = this->positions[base_bin_id + indices[i]];
      }
    }
  }
  delete[] bin_values;

  DensifyHashes(hashes, final_hashes);

  delete[] hashes;
}

void DWTAHashFunction::DensifyHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) {
  // TODO: this could cause exceed max stack size, but is cheaper than memory
  // allocation
  uint32_t* hash_array = new uint32_t[num_hashes]();

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
      if (count > 100) {  // Densification failure.
        break;
      }
    }
    hash_array[i] = next;
  }

  for (uint32_t i = 0; i < L; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < K; j++) {
      uint32_t h = hash_array[i * K + j];
      index += h << ((K - 1 - j) * log_binsize);
    }
    final_hashes[i] = index;
  }
  delete[] hash_array;
}

DWTAHashFunction::~DWTAHashFunction() {
  delete[] bin_map;
  delete[] positions;
}

}  // namespace thirdai::bolt
