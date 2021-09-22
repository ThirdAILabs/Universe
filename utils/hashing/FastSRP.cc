#include "FastSRP.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>

namespace thirdai::utils {

constexpr uint32_t DEFAULT_BINSIZE = 8;

FastSRP::FastSRP(uint32_t input_dim, uint32_t hashes_per_table,
                 uint32_t num_tables, uint32_t seed)
    : HashFunction(num_tables, hashes_per_table),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _dim(input_dim),
      _binsize(DEFAULT_BINSIZE) {
  assert(hashes_per_table < 32);

  _permute = ceil((static_cast<double>(_num_hashes) * _binsize) / _dim);
  _log_num_hashes = log2(_num_hashes);

  std::mt19937 gen(seed);

  uint32_t* n_array = new uint32_t[_dim];
  _bin_map = new uint32_t[_dim * _permute];
  _positions = new uint32_t[_dim * _permute];
  _rand_bits = new uint16_t[_dim * _permute];

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

void FastSRP::hashSingleDense(const float* values, uint32_t dim,
                              uint32_t* output) const {
  // TODO(patrick): this could cause exceed max stack size, but is cheaper than
  // memory allocation
  uint32_t hashes[_num_hashes];
  float bin_values[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < dim; i++) {
      uint32_t binid = _bin_map[base_bin_id + i];
      // if (binid < _num_hashes && bin_values[binid] < data[i]) {
      //   bin_values[binid] = data[i];
      //   hashes[binid] = _positions[base_bin_id + i];
      // }
      if (binid < _num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = values[i] * static_cast<float>(_rand_bits[binid]);
        } else {
          bin_values[binid] +=
              values[i] * static_cast<float>(_rand_bits[binid]);
        }
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }
    }
  }

  densifyHashes(hashes, output);
}

void FastSRP::hashSingleSparse(const uint32_t* indices, const float* values,
                               uint32_t length, uint32_t* output) const {
  // TODO(patrick): this could cause exceed max stack size, but is cheaper than
  // memory allocation
  uint32_t hashes[_num_hashes];
  float bin_values[_num_hashes];

  for (uint32_t i = 0; i < _num_hashes; i++) {
    hashes[i] = std::numeric_limits<uint32_t>::max();
    bin_values[i] = std::numeric_limits<float>::lowest();
  }

  for (uint32_t p = 0; p < _permute; p++) {
    uint32_t base_bin_id = p * _dim;
    for (uint32_t i = 0; i < length; i++) {
      uint32_t binid = _bin_map[base_bin_id + indices[i]];
      // if (binid < _num_hashes && bin_values[binid] < values[i]) {
      //   bin_values[binid] = values[i];
      //   hashes[binid] = _positions[base_bin_id + indices[i]];
      // }
      if (binid < _num_hashes) {
        if (bin_values[binid] == std::numeric_limits<float>::lowest()) {
          bin_values[binid] = values[i] * static_cast<float>(_rand_bits[binid]);
        } else {
          bin_values[binid] +=
              values[i] * static_cast<float>(_rand_bits[binid]);
        }
        hashes[binid] = (bin_values[binid] >= 0 ? 0 : 1);
      }
    }
  }

  densifyHashes(hashes, output);
}

void FastSRP::densifyHashes(const uint32_t* hashes,
                            uint32_t* final_hashes) const {
  // TODO(patrick): this could cause exceed max stack size, but is cheaper than
  // memory allocation
  uint32_t hash_array[_num_hashes];

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
      if (count > 100) {  // Densification failure.
        break;
      }
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

}  // namespace thirdai::utils