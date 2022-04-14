#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "HashFunction.h"
#include <vector>

namespace thirdai::hashing {

class DWTAHashFunction final : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_hashes, _dim, _binsize, _log_binsize,
      _permute;
  std::vector<uint32_t> _bin_map;
  std::vector<uint32_t> _positions;
  uint32_t _rand_double_hash_seed;

  void compactHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HashFunction>(this), _hashes_per_table,
            _num_hashes, _dim, _binsize, _log_binsize, _permute, _bin_map,
            _positions, _rand_double_hash_seed);
  }
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  DWTAHashFunction() : HashFunction(0, 0){};

 public:
  DWTAHashFunction(uint32_t input_dim, uint32_t _hashes_per_table,
                   uint32_t _num_tables, uint32_t range_pow,
                   uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  ~DWTAHashFunction() = default;
};

}  // namespace thirdai::hashing

CEREAL_REGISTER_TYPE(thirdai::hashing::DWTAHashFunction)
