#pragma once

#include "HashFunction.h"

namespace thirdai::utils {

class DWTAHashFunction final : public HashFunction {
 private:
  const uint32_t _hashes_per_table, _num_hashes, _log_2_num_hashes, _dim,
      _binsize, _log_binsize, _permute;
  uint32_t* _bin_map;
  uint32_t* _positions;
  uint32_t _rand_double_hash_seed;

  void compactHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

 public:
  DWTAHashFunction(uint32_t input_dim, uint32_t _hashes_per_table,
                   uint32_t _num_tables, uint32_t range_pow,
                   uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  ~DWTAHashFunction();
};

}  // namespace thirdai::utils
