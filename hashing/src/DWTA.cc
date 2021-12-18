#include "DWTA.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

namespace thirdai::hashing {

constexpr uint32_t DEFAULT_BINSIZE = 8;

DWTAHashFunction::DWTAHashFunction(uint32_t input_dim,
                                   uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t range_pow,
                                   uint32_t seed)
    : HashFunction(num_tables, 1 << range_pow),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _dim(input_dim),
      _binsize(DEFAULT_BINSIZE),
      _log_binsize(floor(log2(_binsize))),
      _permute(ceil((static_cast<double>(_num_hashes) * _binsize) / _dim)) {
  std::mt19937 gen(seed);
  uint32_t* n_array = new uint32_t[_dim];
  _bin_map = new uint32_t[_dim * _permute];
  _positions = new uint32_t[_dim * _permute];

  for (uint32_t i = 0; i < _dim; i++) {
    n_array[i] = i;
  }
  for (uint32_t p = 0; p < _permute; p++) {
    std::shuffle(n_array, n_array + _dim, gen);
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

void DWTAHashFunction::hashSingleDense(const float* values, uint32_t dim,
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
      if (binid < _num_hashes && bin_values[binid] < values[i]) {
        bin_values[binid] = values[i];
        hashes[binid] = _positions[base_bin_id + i];
      }
    }
  }

  delete[] bin_values;

  compactHashes(hashes, output);

  delete[] hashes;
}

void DWTAHashFunction::hashSingleSparse(const uint32_t* indices,
                                        const float* values, uint32_t length,
                                        uint32_t* output) const {
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
      if (binid < _num_hashes && bin_values[binid] < values[i]) {
        bin_values[binid] = values[i];
        hashes[binid] = _positions[base_bin_id + indices[i]];
      }
    }
  }
  delete[] bin_values;

  HashUtils::densifyHashes(hashes, _num_hashes);
  compactHashes(hashes, output);

  delete[] hashes;
}

void DWTAHashFunction::compactHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  // TODO (Josh, Patrick): Figure out how to consolidate this version of
  // compactHashes with HashUtils::defaultCompactHashesMethod.
  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = hashes[i * _hashes_per_table + j];
      index += h << ((_hashes_per_table - 1 - j) * _log_binsize);
    }
    final_hashes[i] = index;
  }
}

DWTAHashFunction::~DWTAHashFunction() {
  delete[] _bin_map;
  delete[] _positions;
}

}  // namespace thirdai::hashing