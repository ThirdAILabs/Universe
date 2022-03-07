#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "HashFunction.h"
#include <cstdint>

namespace thirdai::hashing {

class FastSRP : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_hashes, _log_num_hashes, _dim, _binsize,
      _permute, _rand_double_hash_seed;
  std::vector<uint32_t> _bin_map, _positions;
  std::vector<int8_t> _rand_bits;

  FastSRP() : HashFunction(0, 0){};
  friend class cereal::access;

 public:
  FastSRP(uint32_t input_dim, uint32_t _hashes_per_table, uint32_t _num_tables,
          uint32_t out_mod = UINT32_MAX, uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  // This method lets cereal know which data members to serialize
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HashFunction>(this), _hashes_per_table,
            _num_hashes, _log_num_hashes, _dim, _binsize, _permute,
            _rand_double_hash_seed, _bin_map, _positions, _rand_bits);
  }
};

}  // namespace thirdai::hashing

CEREAL_REGISTER_TYPE(thirdai::hashing::FastSRP);
