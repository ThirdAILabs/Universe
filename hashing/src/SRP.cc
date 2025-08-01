#include "SRP.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

namespace thirdai::hashing {

// TODO(Any). This class is currently untested (though it is also not clear
// is is useful, so at some point we may delete it).
SignedRandomProjection::SignedRandomProjection(uint32_t input_dim,
                                               uint32_t srps_per_table,
                                               uint32_t num_tables,
                                               uint32_t seed)
    : HashFunction(num_tables, 1 << srps_per_table),
      _srps_per_table(srps_per_table),
      _total_num_srps(srps_per_table * num_tables),
      _dim(input_dim),
      _sample_size(ceil(_dim * 0.3)) {
  assert(srps_per_table < 32);

  uint32_t* a = new uint32_t[_dim];
  for (uint32_t i = 0; i < _dim; i++) {
    a[i] = i;
  }

  std::mt19937 gen(seed);

  _random_bits = new int16_t[_total_num_srps * _sample_size];
  _hash_indices = new uint32_t[_total_num_srps * _sample_size];

  for (uint32_t i = 0; i < _total_num_srps; i++) {
    std::shuffle(a, a + _dim, gen);
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

void SignedRandomProjection::hashSingleDense(const float* values, uint32_t dim,
                                             uint32_t* output) const {
  assert(dim == _dim);
  (void)dim;

  memset(output, 0, _num_tables * sizeof(uint32_t));

  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint32_t srp = 0; srp < _srps_per_table; srp++) {
      double s = 0;
      for (uint32_t srp_part = 0; srp_part < _sample_size; srp_part++) {
        uint32_t bit_index = table * _srps_per_table * _sample_size +
                             srp * _sample_size + srp_part;
        s += static_cast<float>(_random_bits[bit_index]) *
             values[_hash_indices[bit_index]];
      }
      uint32_t to_add = (s > 0) << srp;
      output[table] += to_add;
    }
  }
}

void SignedRandomProjection::hashSingleSparse(const uint32_t* indices,
                                              const float* values,
                                              uint32_t length,
                                              uint32_t* output) const {
  memset(output, 0, _num_tables * sizeof(uint32_t));

  for (uint32_t table = 0; table < _num_tables; table++) {
    for (uint32_t srp = 0; srp < _srps_per_table; srp++) {
      double s = 0;
      uint32_t indices_index = 0;

      for (uint32_t srp_part = 0; srp_part < _sample_size; srp_part++) {
        uint32_t bit_index = table * _srps_per_table * _sample_size +
                             srp * _sample_size + srp_part;
        uint32_t hash_index = _hash_indices[bit_index];
        while (indices_index < length && hash_index > indices[indices_index]) {
          indices_index++;
        }
        if (indices_index < length && hash_index == indices[indices_index]) {
          s += static_cast<float>(_random_bits[bit_index]) *
               values[indices_index];
        }
      }
      uint32_t to_add = (s > 0) << srp;
      output[table] += to_add;
    }
  }
}

SignedRandomProjection::~SignedRandomProjection() {
  delete[] _random_bits;
  delete[] _hash_indices;
}

}  // namespace thirdai::hashing
