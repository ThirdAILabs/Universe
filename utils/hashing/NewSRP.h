#pragma once

#include <stdint.h>

namespace bolt {

class FastSRP {
 private:
  uint32_t _num_hashes_per_table, _num_tables, _num_hashes, _log_num_hashes,
      _dim, _binsize, _permute;
  uint32_t* _bin_map;
  uint32_t* _positions;
  uint32_t _rand_double_hash_seed;
  short* _rand_bits;

  constexpr uint32_t RandDoubleHash(int binid, int count) {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  void DensifyHashes(uint32_t* hashes, uint32_t* final_hashes);

 public:
  FastSRP(uint32_t input_dim, uint32_t _num_hashes_per_table,
          uint32_t _num_tables, uint32_t range_pow);

  uint32_t* HashSparseVector(const uint32_t* indices, const float* values,
                             uint32_t len);

  void HashSparseVector(const uint32_t* indices, const float* values,
                        uint32_t len, uint32_t* final_hashes);

  uint32_t* HashVector(const float* data, uint32_t len);

  void HashVector(const float* data, uint32_t len, uint32_t* final_hashes);

  ~FastSRP();
};

}  // namespace bolt