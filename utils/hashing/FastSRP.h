#pragma once

#include "HashFunction.h"
#include <cstdint>

namespace thirdai::utils {

class FastSRP : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_hashes, _log_num_hashes, _dim, _binsize,
      _permute;
  uint32_t* _bin_map;
  uint32_t* _positions;
  uint32_t _rand_double_hash_seed;
  uint16_t* _rand_bits;

  constexpr uint32_t RandDoubleHash(int binid, int count) const {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  void densifyHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

 public:
  FastSRP(uint32_t input_dim, uint32_t _hashes_per_table, uint32_t _num_tables,
          uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  ~FastSRP();
};

}  // namespace thirdai::utils