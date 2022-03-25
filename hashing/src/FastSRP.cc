#include "FastSRP.h"
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

namespace thirdai::hashing {

constexpr uint32_t DEFAULT_BINSIZE = 8;

FastSRP::FastSRP(uint32_t input_dim, uint32_t hashes_per_table,
                 uint32_t num_tables, uint32_t out_mod, uint32_t seed)
    : HashFunction(num_tables,
                   std::min<uint32_t>(out_mod, 1 << hashes_per_table)),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _log_num_hashes(log2(_num_hashes)),
      _dim(input_dim),
      _binsize(DEFAULT_BINSIZE),
      _permute(ceil((static_cast<double>(_num_hashes) * _binsize) / _dim)),
      _bin_map(_dim * _permute),
      _positions(_dim * _permute),
      _rand_bits(_dim * _permute) {
  if (_hashes_per_table >= 32) {
    throw std::invalid_argument("For now, we require <31 SRP per hash");
  }

  std::mt19937 gen(seed);
  _rand_double_hash_seed = gen();

  std::vector<uint32_t> n_array(_dim);
  for (uint32_t i = 0; i < _dim; i++) {
    n_array[i] = i;
  }

  for (uint32_t i = 0; i < _dim * _permute; i++) {
    uint32_t curr = gen();
    if (curr % 2 == 0) {
      _rand_bits[i] = 1;
    } else {
      _rand_bits[i] = -1;
    }
  }

  for (uint32_t p = 0; p < _permute; p++) {
    std::shuffle(n_array.begin(), n_array.end(), gen);
    for (uint32_t j = 0; j < _dim; j++) {
      _bin_map[p * _dim + n_array[j]] = (p * _dim + j) / _binsize;
      _positions[p * _dim + n_array[j]] = (p * _dim + j) % _binsize;
    }
  }
}

void FastSRP::hashSingleDense(const float* values, uint32_t dim,
                              uint32_t* output) const {
  uint32_t* hashes = new uint32_t[_num_hashes];
  float* bin_values = new float[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < dim; i++) {
      uint32_t binid = _bin_map[base_bin_id + i];
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
  delete[] bin_values;

  // TODO(Josh, Patrick): Shouldn't we densify here too?
  HashUtils::compactHashBits(hashes, output, _num_tables, _hashes_per_table);
  for (uint32_t i = 0; i < _num_tables; i++) {
    output[i] %= _range;
  }
  delete[] hashes;
}

void FastSRP::hashSingleSparse(const uint32_t* indices, const float* values,
                               uint32_t length, uint32_t* output) const {
  // TODO(josh, Patrick): Should we assert indices are within the expected
  // dimension here (otherwise it's a segfault)
  uint32_t* hashes = new uint32_t[_num_hashes];
  float* bin_values = new float[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < length; i++) {
      uint32_t binid = _bin_map[base_bin_id + indices[i]];
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

  delete[] bin_values;

  HashUtils::densifyHashes(hashes, _num_hashes);
  HashUtils::compactHashBits(hashes, output, _num_tables, _hashes_per_table);
  for (uint32_t i = 0; i < _num_tables; i++) {
    output[i] %= _range;
  }

  delete[] hashes;
}

}  // namespace thirdai::hashing