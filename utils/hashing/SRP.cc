#include "SRP.h"
// #include "config.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdint.h>

namespace bolt {

SparseRandomProjection::SparseRandomProjection(uint32_t input_dim, uint32_t K,
                                               uint32_t L, uint32_t range_pow)
    : _K(K), _L(L), _num_hashes(K * L), _range(1 << range_pow), _dim(input_dim) {
  _sample_size = ceil(1.0 * _dim / _ratio);

  uint32_t* a = new uint32_t[_dim];
  for (uint32_t i = 0; i < _dim; i++) {
    a[i] = i;
  }

  srand(time(0));

  _random_bits = new short*[_num_hashes];
  _hash_indices = new uint32_t*[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    std::random_shuffle(a, a + _dim);
    _random_bits[i] = new short[_sample_size];
    _hash_indices[i] = new uint32_t[_sample_size];
    for (uint32_t j = 0; j < _sample_size; j++) {
      _hash_indices[i][j] = a[j];
      uint32_t curr = rand();
      if (curr % 2 == 0) {
        _random_bits[i][j] = 1;
      } else {
        _random_bits[i][j] = -1;
      }
    }
    std::sort(_hash_indices[i], _hash_indices[i] + _sample_size);
  }
  delete[] a;
}

void SparseRandomProjection::HashVector(const float* data, uint32_t len,
                                        uint32_t* final_hashes) {
  // length should be = to this->_dim
  uint32_t hashes[_num_hashes];

  // #pragma omp parallel for
  for (uint32_t i = 0; i < _num_hashes; i++) {
    double s = 0;
    for (uint32_t j = 0; j < _sample_size; j++) {
      float v = data[_hash_indices[i][j]];
      if (_random_bits[i][j] >= 0) {
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
  uint32_t* final_hashes = new uint32_t[_L];
  HashVector(data, len, final_hashes);
  return final_hashes;
}

void SparseRandomProjection::HashSparseVector(const uint32_t* indices,
                                              const float* values, uint32_t len,
                                              uint32_t* final_hashes) {
  uint32_t hashes[_num_hashes];

  for (uint32_t p = 0; p < _num_hashes; p++) {
    double s = 0;
    uint32_t i = 0;
    uint32_t j = 0;
    while ((i < len) & (j < _sample_size)) {
      if (indices[i] == _hash_indices[p][j]) {
        float v = values[i];
        if (_random_bits[p][j] >= 0) {
          s += v;
        } else {
          s -= v;
        }
        i++;
        j++;
      } else if (indices[i] < _hash_indices[p][j]) {
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
  uint32_t* final_hashes = new uint32_t[_L];
  HashSparseVector(indices, values, len, final_hashes);
  return final_hashes;
}

void SparseRandomProjection::CompactHashes(uint32_t* hashes,
                                           uint32_t* final_hashes) {
  for (uint32_t i = 0; i < _L; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _K; j++) {
      uint32_t h = hashes[_K * i + j];
      index += h << (_K - 1 - j);
    }

    final_hashes[i] = index % _range;
  }
}

SparseRandomProjection::~SparseRandomProjection() {
  for (uint32_t i = 0; i < _num_hashes; i++) {
    delete[] _random_bits[i];
    delete[] _hash_indices[i];
  }
  delete[] _random_bits;
  delete[] _hash_indices;
}

}  // namespace bolt
