// #pragma once

#include "HashFunction.h"
#include "MurmurHash.h"
#include <stdint.h>

namespace thirdai::utils {

class DensifiedMinHash : public HashFunction {
 private:
  uint32_t _hashes_per_table, _num_tables, _num_hashes, _range, _binsize;
  uint32_t _randa, _rand_double_hash_seed;
  uint32_t *_random_hash, *_rand1, _log_num_hashes, *_binids;
  uint32_t _topK = 30;

  void densifyHashes(const uint32_t* hashes, uint32_t* final_hashes) const;

  void hashSparseVector(const uint32_t* indices, const float* values,
                        uint32_t len, uint32_t* final_hashes) const;

  void hashDenseVector(const float* data, uint32_t len,
                       uint32_t* final_hashes) const;

 public:
  constexpr uint32_t RandDoubleHash(int binid, int count) const {
    uint32_t tohash = ((binid + 1) << 6) + count;
    uint32_t result =
        (_rand_double_hash_seed * tohash << 3) >> (32 - _log_num_hashes);
    return result;
  }

  DensifiedMinHash(uint32_t input_dim, uint32_t hashes_per_table,
                   uint32_t num_tables, uint32_t range_pow, uint32_t seed);

  void hashSparse(uint64_t num_vectors, const uint32_t* const* indices,
                  const float* const* values, const uint32_t* lengths,
                  uint32_t* output) const override;

  void hashDense(uint64_t num_vectors, uint64_t dim, const float* const* values,
                 uint32_t* output) const override;

  uint32_t numTables() const override { return _num_tables; }

  uint32_t range() const override { return _range; }

  ~DensifiedMinHash();
};

}  // namespace thirdai::utils
