// #pragma once

#include "HashFunction.h"
#include "MurmurHash.h"
#include <cstdint>

namespace thirdai::utils {

class DensifiedMinHash : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_hashes, _binsize;
  uint32_t _randa, _rand_double_hash_seed;
  uint32_t *_random_hash, *_rand1, _log_num_hashes, *_binids;
  uint32_t _topK = 30;

  void densifyHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

 public:
  constexpr uint32_t RandDoubleHash(int binid, int count) const {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  DensifiedMinHash(uint32_t input_dim, uint32_t hashes_per_table,
                   uint32_t num_tables, uint32_t range_pow, uint32_t seed);

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  ~DensifiedMinHash();
};

}  // namespace thirdai::utils
