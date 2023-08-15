#include "DWTA.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <proto/hashing.pb.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <random>

namespace thirdai::hashing {

DWTAHashFunction::DWTAHashFunction(uint32_t input_dim,
                                   uint32_t hashes_per_table,
                                   uint32_t num_tables, uint32_t range_pow,
                                   uint32_t binsize,
                                   std::optional<uint32_t> permutations,
                                   uint32_t seed)
    : HashFunction(num_tables, 1 << range_pow),
      _hashes_per_table(hashes_per_table),
      _num_hashes(hashes_per_table * num_tables),
      _dim(input_dim),
      _binsize(binsize),
      _log_binsize(floor(log2(_binsize))) {
  _permute = permutations.value_or(
      ceil((static_cast<double>(_num_hashes) * _binsize) / _dim));

  std::mt19937 gen(seed);
  _bin_map = std::vector<uint32_t>(_dim * _permute);
  _positions = std::vector<uint32_t>(_dim * _permute);

  uint32_t n_bin_locs = _num_hashes * _binsize;
  uint32_t n_rounds = (_dim * _permute + n_bin_locs - 1) / n_bin_locs;

  std::vector<uint32_t> bin_locs(n_rounds * n_bin_locs);

  for (size_t i = 0; i < n_rounds; i++) {
    auto start = bin_locs.begin() + i * n_bin_locs;
    auto end = bin_locs.begin() + (i + 1) * n_bin_locs;
    std::iota(start, end, 0);
    if (i == n_rounds - 1) {
      // Shuffle the last set of bin locations so that if all of them are not
      // used, it is not biased towards lower bin locations.
      std::shuffle(start, end, gen);
    }
  }

  std::shuffle(bin_locs.begin(), bin_locs.begin() + _dim * _permute, gen);

  for (size_t i = 0; i < _permute * _dim; i++) {
    _bin_map[i] = bin_locs[i] / _binsize;
    _positions[i] = bin_locs[i] % _binsize;
  }

  std::uniform_int_distribution<uint32_t> dis(
      1, std::numeric_limits<uint32_t>::max() - 1);

  _rand_double_hash_seed = dis(gen);
}

DWTAHashFunction::DWTAHashFunction(const proto::hashing::DWTA& dwta_proto)
    : HashFunction(
          dwta_proto.num_tables(),
          1 << (dwta_proto.hashes_per_table() * dwta_proto.log_binsize())),
      _hashes_per_table(dwta_proto.hashes_per_table()),
      _num_hashes(dwta_proto.num_tables() * dwta_proto.hashes_per_table()),
      _dim(dwta_proto.input_dim()),
      _binsize(1 << dwta_proto.log_binsize()),
      _log_binsize(dwta_proto.log_binsize()),
      _permute(dwta_proto.permutations()),
      _bin_map(dwta_proto.bin_map().begin(), dwta_proto.bin_map().end()),
      _positions(dwta_proto.positions().begin(), dwta_proto.positions().end()),
      _rand_double_hash_seed(dwta_proto.random_double_hash_seed()) {}

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

proto::hashing::HashFunction* DWTAHashFunction::toProto() const {
  proto::hashing::HashFunction* hash_fn = new proto::hashing::HashFunction();

  auto* dwta = hash_fn->mutable_dwta();
  dwta->set_num_tables(_num_tables);
  dwta->set_hashes_per_table(_hashes_per_table);
  dwta->set_log_binsize(_log_binsize);
  dwta->set_input_dim(_dim);
  dwta->set_permutations(_permute);

  *dwta->mutable_bin_map() = {_bin_map.begin(), _bin_map.end()};
  *dwta->mutable_positions() = {_positions.begin(), _positions.end()};

  dwta->set_random_double_hash_seed(_rand_double_hash_seed);

  return hash_fn;
}

void DWTAHashFunction::save(const std::string& filename) const {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output_stream);
}

void DWTAHashFunction::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<DWTAHashFunction> DWTAHashFunction::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<DWTAHashFunction> DWTAHashFunction::load_stream(
    std::istream& input_stream) {
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
