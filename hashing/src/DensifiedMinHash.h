#pragma once

#include "HashFunction.h"
#include "MurmurHash.h"
#include <cstdint>

namespace thirdai::hashing {

/** Based off of the paper https://arxiv.org/pdf/1703.04664.pdf */
class DensifiedMinHash final : public HashFunction {
 public:
  // TODO(josh): Remove range when we have the hash function wrappers done
  DensifiedMinHash(uint32_t hashes_per_table, uint32_t num_tables,
                   uint32_t range, uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<DensifiedMinHash>(
        /* hashes_per_table= */ _hashes_per_table,
        /* num_tables= */ _num_tables,
        /* range= */ _range);
  }

 private:
  const uint32_t _hashes_per_table, _total_num_hashes, _binsize, _seed;
};

}  // namespace thirdai::hashing
