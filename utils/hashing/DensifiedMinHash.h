// #pragma once

#include "HashFunction.h"
#include "MurmurHash.h"
#include <cstdint>

namespace thirdai::utils {

/** Based off of the paper https://arxiv.org/pdf/1703.04664.pdf */
class DensifiedMinHash : public HashFunction {
 public:
  DensifiedMinHash(uint32_t hashes_per_table, uint32_t num_tables,
                   uint32_t seed);

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

 private:
  const uint32_t _hashes_per_table, _total_num_hashes, _binsize, _seed;

  void compactHashes(const uint32_t* hashes, uint32_t* final_hashes) const;
};

}  // namespace thirdai::utils
