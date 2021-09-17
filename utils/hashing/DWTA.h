#pragma once

#include "HashFunction.h"

namespace thirdai::utils {

class DWTAHashFunction final : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_tables, _num_hashes, _log_num_hashes, _range,
      _dim, _binsize, _log_binsize, _permute;
  uint32_t* _bin_map;
  uint32_t* _positions;
  uint32_t _rand_double_hash_seed;

  constexpr uint32_t RandDoubleHash(uint32_t binid, uint32_t count) const {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  void densifyHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

  void hashSparseVector(const uint32_t* indices, const float* values,
                        uint32_t len, uint32_t* final_hashes) const;

  void hashDenseVector(const float* data, uint32_t len,
                       uint32_t* final_hashes) const;

 public:
  DWTAHashFunction(uint32_t input_dim, uint32_t _hashes_per_table,
                   uint32_t _num_tables, uint32_t range_pow,
                   uint32_t seed = time(nullptr));

  void hashSparse(uint64_t num_vectors, const uint32_t* const* indices,
                  const float* const* values, const uint32_t* lengths,
                  uint32_t* output) const override;

  void hashDense(uint64_t num_vectors, uint64_t dim, const float* const* values,
                 uint32_t* output) const override;

  uint32_t numTables() const { return _num_tables; }

  uint32_t range() const { return _range; }

  ~DWTAHashFunction();
};

}  // namespace thirdai::utils
