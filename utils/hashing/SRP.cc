#include "SRP.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>

namespace thirdai::utils {

SparseRandomProjection::SparseRandomProjection(uint32_t input_dim,
                                               uint32_t srps_per_table,
                                               uint32_t num_tables,
                                               uint32_t range_pow)
    : _srps_per_table(srps_per_table),
      _num_tables(num_tables),
      _total_num_srps(srps_per_table * num_tables),
      _range(1 << range_pow),
      _dim(input_dim),
      _sample_size(ceil(_dim * _ratio)) {
  assert(srps_per_table < 32);

  uint32_t* a = new uint32_t[_dim];
  for (uint32_t i = 0; i < _dim; i++) {
    a[i] = i;
  }

  // TODO(alan, patrick): Fix global seed.
  srand(time(0));

  _random_bits = new short[_total_num_srps * _sample_size];
  _hash_indices = new uint32_t[_total_num_srps * _sample_size];

  for (uint32_t i = 0; i < _total_num_srps; i++) {
    std::shuffle(a, a + _dim, std::default_random_engine(rand()));
    for (uint32_t j = 0; j < _sample_size; j++) {
      _hash_indices[i * _sample_size + j] = a[j];
      uint32_t curr = rand();
      _random_bits[i * _sample_size + j] = (curr % 2) * 2 - 1;
    }
    std::sort(_hash_indices + i * _sample_size,
              _hash_indices + (i + 1) * _sample_size);
  }
  delete[] a;
}

void SparseRandomProjection::hashDense(uint64_t num_vectors, uint64_t dim,
                                       float** values, uint32_t* output) const {
  assert(dim == _dim);

  memset(output, 0, _num_tables);

// TODO: Is this the loop we want parallelism on? I(Josh) think yes
#pragma omp parallel for default(none) shared(num_vectors, values, output)
  for (uint32_t vec = 0; vec < num_vectors; vec++) {
    for (uint32_t table = 0; table < _num_tables; table++) {
      for (uint32_t srp = 0; srp < _srps_per_table; srp++) {
        double s = 0;
        for (uint32_t srp_part = 0; srp_part < _sample_size; srp_part++) {
          uint32_t bit_index = table * _srps_per_table * _sample_size +
                               srp * _sample_size + srp_part;
          s += _random_bits[bit_index] * values[vec][_hash_indices[bit_index]];
        }
        uint32_t to_add = (s > 0) << srp;
        output[table] += to_add;
      }
    }
  }
}


void SparseRandomProjection::hashSparse(uint64_t num_vectors,
                                        uint32_t** indices, float** values,
                                        uint32_t* lengths, uint32_t* output) const {
#pragma omp parallel for default(none) \
    shared(indices, values, lengths, output, num_vectors)
  for (uint32_t vec = 0; vec < num_vectors; vec++) {
    for (uint32_t table = 0; table < _num_tables; table++) {
      for (uint32_t srp = 0; srp < _srps_per_table; srp++) {
        
        double s = 0;
        
        uint32_t* current_indices = indices[vec];
        uint32_t indices_index = 0;
        float* current_vals = values[vec];
        uint32_t current_length = lengths[vec];

        for (uint32_t srp_part = 0; srp_part < _sample_size; srp_part++) {
          uint32_t bit_index = table * _srps_per_table * _sample_size +
                               srp * _sample_size + srp_part;
          uint32_t hash_index = _hash_indices[bit_index];
          while (indices_index < current_length && hash_index > current_indices[indices_index]) {
            indices_index++;
          }
          if (indices_index < current_length && hash_index == current_indices[indices_index]) {
            s += _random_bits[bit_index] * current_vals[indices_index];
          }
        }
        uint32_t to_add = (s > 0) << srp;
        output[table] += to_add;
      }
    }
  }
}

SparseRandomProjection::~SparseRandomProjection() {
  delete[] _random_bits;
  delete[] _hash_indices;
}

}  // namespace thirdai::utils
