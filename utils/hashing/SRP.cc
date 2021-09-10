#include "SRP.h"
// #include "config.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace thirdai::utils {

SparseRandomProjection::SparseRandomProjection(uint32_t input_dim, uint32_t hashes_per_table,
                                               uint32_t num_tables, uint32_t range_pow)
    : _hashes_per_table(hashes_per_table),
      _num_tables(num_tables),
      _num_hashes(hashes_per_table * num_tables),
      _range(1 << range_pow),
      _dim(input_dim) {
  _sample_size = ceil(1.0 * _dim / _ratio);

  uint32_t* a = new uint32_t[_dim];
  for (uint32_t i = 0; i < _dim; i++) {
    a[i] = i;
  }

  // TODO: Fix global seed.
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

void SparseRandomProjection::hashDense(uint64_t num_vectors, uint64_t dim,
                                       float** values, uint32_t num_hashes,
                                       uint32_t* output) {
  for (uint32_t i = 0; i < num_vectors; i++) {
    SparseRandomProjection::hashDenseVector(i, values, num_hashes,
                                            output + i * _num_tables);
  }
  (void)dim;
}

void SparseRandomProjection::hashDenseVector(uint32_t index, float** values,
                                             uint32_t num_hashes,
                                             uint32_t* output) {
  // length should be = to this->_dim
  uint32_t hashes[_num_hashes];

  // #pragma omp parallel for
  for (uint32_t i = 0; i < _num_hashes; i++) {
    double s = 0;
    for (uint32_t j = 0; j < _sample_size; j++) {
      float v = values[index][_hash_indices[i][j]];
      if (_random_bits[i][j] >= 0) {
        s += v;
      } else {
        s -= v;
      }
    }
    hashes[i] = (s >= 0 ? 0 : 1);
  }
  CompactHashes(hashes, output + index * _num_tables);
  (void)num_hashes;
}

void SparseRandomProjection::hashSparse(uint64_t num_vectors,
                                        uint32_t** indices, float** values,
                                        uint32_t* lengths, uint64_t num_hashes,
                                        uint32_t* output) {
  for (uint32_t i = 0; i < num_vectors; i++) {
    SparseRandomProjection::hashSparseVector(i, indices, values, lengths,
                                             num_hashes, output + i * _num_tables);
  }
}
void SparseRandomProjection::hashSparseVector(
    uint32_t index, uint32_t** indices, float** values, const uint32_t* lengths,
    uint64_t num_hashes, uint32_t* output) {
  uint32_t hashes[_num_hashes];

  for (uint32_t p = 0; p < _num_hashes; p++) {
    double s = 0;
    uint32_t i = 0;
    uint32_t j = 0;
    while ((i < *lengths) & (j < _sample_size)) {
      if (indices[index][i] == _hash_indices[p][j]) {
        float v = values[index][i];
        if (_random_bits[p][j] >= 0) {
          s += v;
        } else {
          s -= v;
        }
        i++;
        j++;
      } else if (indices[index][i] < _hash_indices[p][j]) {
        i++;
      } else {
        j++;
      }
    }
    hashes[p] = (s >= 0 ? 0 : 1);
  }

  CompactHashes(hashes, output + index * _num_tables);
  (void)num_hashes;
}

void SparseRandomProjection::CompactHashes(uint32_t* hashes,
                                           uint32_t* final_hashes) {
  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = hashes[_hashes_per_table * i + j];
      index += h << (_hashes_per_table - 1 - j);
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

}  // namespace thirdai::utils
