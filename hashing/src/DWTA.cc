#include "DWTA.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <algorithm>
#include <limits>
#include <random>
#include <unordered_map>

namespace thirdai::hashing {

DWTAHashFunction::DWTAHashFunction(uint32_t input_dim,
                                   uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t range_pow,
                                   uint32_t binsize, uint32_t permutations,
                                   uint32_t seed)
    : HashFunction(num_tables, 1 << range_pow),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _dim(input_dim),
      _binsize(binsize),
      _log_binsize(floor(log2(_binsize))),
      _permute(permutations) {
  std::mt19937 gen(seed);
  _bin_map = std::vector<uint32_t>(_dim * _permute);
  _positions = std::vector<uint32_t>(_dim * _permute);

  uint32_t* n_array = new uint32_t[_dim];
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

  densifyHashes(hashes, _num_hashes);
  compactHashes(hashes, output);

  delete[] hashes;
}

void DWTAHashFunction::compactHashes(const uint32_t* hashes,
                                     uint32_t* final_hashes) const {
  for (uint32_t i = 0; i < _num_tables; i++) {
    uint32_t index = 0;
    for (uint32_t j = 0; j < _hashes_per_table; j++) {
      uint32_t h = hashes[i * _hashes_per_table + j];

      /**
       * This is to fix a suble bug caused by NaNs. When hashing vector of NaNs
       * the value of a NaN does not exceed the lowest float value and so the
       * hashes are never overwritten from 2^32-1. This bit shift clears all but
       * the lower order log(binsize) bits so that each hash is in the range
       * [0, binsize) and only has log(binsize) bits. Then when these hashes are
       * concatenated the output hash index has the correct number of bits for
       * the output range: num_hashes_per_table * log(binsize).
       */
      h = h & (_binsize - 1);

      index += h << ((_hashes_per_table - 1 - j) * _log_binsize);
    }
    final_hashes[i] = index;
  }
}

void DWTAHashFunction::save(const std::string& filename) {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<HashFunction> DWTAHashFunction::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<DWTAHashFunction> deserialize_into(new DWTAHashFunction());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template <class Archive>
void DWTAHashFunction::serialize(Archive& archive) {
  archive(cereal::base_class<HashFunction>(this), _hashes_per_table,
          _num_hashes, _dim, _binsize, _log_binsize, _permute, _bin_map,
          _positions, _rand_double_hash_seed);
}

template void DWTAHashFunction::serialize(cereal::PortableBinaryInputArchive&);
template void DWTAHashFunction::serialize(cereal::PortableBinaryOutputArchive&);

template void DWTAHashFunction::serialize(cereal::BinaryInputArchive&);
template void DWTAHashFunction::serialize(cereal::BinaryOutputArchive&);

}  // namespace thirdai::hashing

CEREAL_REGISTER_TYPE(thirdai::hashing::DWTAHashFunction)
