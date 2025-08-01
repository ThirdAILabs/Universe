#pragma once

#include "HashFunction.h"
#include <utils/Random.h>
#include <cstdint>

namespace thirdai::hashing {

class FastSRP final : public HashFunction {
 public:
  FastSRP(uint32_t input_dim, uint32_t _hashes_per_table, uint32_t _num_tables,
          uint32_t out_mod = UINT32_MAX,
          uint32_t seed = global_random::nextSeed());

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<FastSRP>(
        /* input_dim= */ _dim, /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables, /* out_mod= */ _range);
  }
  std::string getName() const final { return "FastSRP"; }

 private:
  uint32_t _hashes_per_table, _num_hashes, _log_num_hashes, _dim, _binsize,
      _permute, _rand_double_hash_seed;
  std::vector<uint32_t> _bin_map, _positions;
  std::vector<int8_t> _rand_bits;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  FastSRP() : HashFunction(0, 0){};
};

}  // namespace thirdai::hashing
