// #pragma once

#include "MurmurHash.h"
#include <stdint.h>
//#include "Hash_Function.h"

namespace bolt {

class DensifiedMinHash {
 private:
  uint32_t _K, _L, _num_hashes, _range, _binsize;
  uint32_t _randa, _rand_double_hash_seed;
  uint32_t *_random_hash, *_rand1, _log_num_hashes, *_binids;
  uint32_t _topK = 30;

 public:
  constexpr uint32_t RandDoubleHash(int binid, int count) {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  DensifiedMinHash(uint32_t input_dim, uint32_t _K, uint32_t _L,
                   uint32_t range_pow);

  uint32_t* HashSparseVector(const uint32_t* indices, const float* values,
                             uint32_t len);

  void HashSparseVector(const uint32_t* indices, const float* values,
                        uint32_t len, uint32_t* final_hashes);

  uint32_t* HashVector(const float* data, uint32_t len);

  void HashVector(const float* data, uint32_t len, uint32_t* final_hashes);

  void DensifyHashes(uint32_t* hashes, uint32_t* final_hashes);

  void getMap(int n, int* binid);
  ~DensifiedMinHash();
};

}  // namespace bolt
