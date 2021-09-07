#include "SRP.h"
// #include "config.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdint.h>

namespace bolt {

SparseRandomProjection::SparseRandomProjection(uint32_t input_dim, uint32_t _K,
                                               uint32_t _L, uint32_t range_pow)
    : K(_K), L(_L), num_hashes(K * L), range(1 << range_pow), dim(input_dim) {
  sample_size = ceil(1.0 * dim / ratio);

  uint32_t* a = new uint32_t[dim];
  for (uint32_t i = 0; i < dim; i++) {
    a[i] = i;
  }

  srand(time(0));

  random_bits = new short*[num_hashes];
  hash_indices = new uint32_t*[num_hashes];

  for (uint32_t i = 0; i < num_hashes; i++) {
    std::random_shuffle(a, a + dim);
    random_bits[i] = new short[sample_size];
    hash_indices[i] = new uint32_t[sample_size];
    for (uint32_t j = 0; j < sample_size; j++) {
      hash_indices[i][j] = a[j];
      uint32_t curr = rand();
      if (curr % 2 == 0) {
        random_bits[i][j] = 1;
      } else {
        random_bits[i][j] = -1;
      }
    }
    std::sort(hash_indices[i], hash_indices[i] + sample_size);
  }
  delete[] a;
}

void SparseRandomProjection::HashVector(const float* data, uint32_t len,
                                        uint32_t* final_hashes) {
  // length should be = to this->dim
  uint32_t hashes[num_hashes];

  // #pragma omp parallel for
  for (uint32_t i = 0; i < num_hashes; i++) {
    double s = 0;
    for (uint32_t j = 0; j < sample_size; j++) {
      float v = data[hash_indices[i][j]];
      if (random_bits[i][j] >= 0) {
        s += v;
      } else {
        s -= v;
      }
    }
    hashes[i] = (s >= 0 ? 0 : 1);
  }

  CompactHashes(hashes, final_hashes);
  (void)len;
}

uint32_t* SparseRandomProjection::HashVector(const float* data, uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void SparseRandomProjection::HashSparseVector(const uint32_t* indices,
                                              const float* values, uint32_t len,
                                              uint32_t* final_hashes) {
  uint32_t hashes[num_hashes];

  for (uint32_t p = 0; p < num_hashes; p++) {
    double s = 0;
    uint32_t i = 0;
    uint32_t j = 0;
    while ((i < len) & (j < sample_size)) {
      if (indices[i] == hash_indices[p][j]) {
        float v = values[i];
        if (random_bits[p][j] >= 0) {
          s += v;
        } else {
          s -= v;
        }
        i++;
        j++;
      } else if (indices[i] < hash_indices[p][j]) {
        i++;
      } else {
        j++;
      }
    }
    hashes[p] = (s >= 0 ? 0 : 1);
  }

  CompactHashes(hashes, final_hashes);
}

uint32_t* SparseRandomProjection::HashSparseVector(const uint32_t* indices,
                                                   const float* values,
                                                   uint32_t len) {
  uint32_t* final_hashes = new uint32_t[L];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void SparseRandomProjection::CompactHashes(uint32_t* hashes,
                                           uint32_t* final_hashes) {
  for (uint32_t i = 0; i < L; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < K; j++) {
      uint32_t h = hashes[K * i + j];
      index += h << (K - 1 - j);
    }

    final_hashes[i] = index % range;
  }
}

SparseRandomProjection::~SparseRandomProjection() {
  for (uint32_t i = 0; i < num_hashes; i++) {
    delete[] random_bits[i];
    delete[] hash_indices[i];
  }
  delete[] random_bits;
  delete[] hash_indices;
}

}  // namespace bolt
