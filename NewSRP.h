#pragma once

#include <stdint.h>

namespace bolt {

class FastSRP {
 private:
  uint32_t K, L, num_hashes, log_num_hashes, range, dim, binsize, log_binsize,
      permute;
  uint32_t* bin_map;
  uint32_t* positions;
  uint32_t rand_double_hash_seed;
  short* rand_bits;

  constexpr uint32_t RandDoubleHash(int binid, int count) {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (rand_double_hash_seed * tohash << 3) >> (32 - this->log_num_hashes);
    return result;
  }

  void DensifyHashes(uint32_t* hashes, uint32_t* final_hashes);

 public:
  FastSRP(uint32_t input_dim, uint32_t _K, uint32_t _L, uint32_t range_pow);

  uint32_t* HashSparseVector(const uint32_t* indices, const float* values,
                             uint32_t len);

  void HashSparseVector(const uint32_t* indices, const float* values,
                        uint32_t len, uint32_t* final_hashes);

  uint32_t* HashVector(const float* data, uint32_t len);

  void HashVector(const float* data, uint32_t len, uint32_t* final_hashes);

  ~FastSRP();
};

}  // namespace bolt