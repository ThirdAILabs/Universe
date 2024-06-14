#pragma once

#include "HashFunction.h"
#include <utils/Random.h>
#include <cstdint>
#include <vector>

namespace thirdai::hashing {

class SignedRandomProjection final : public HashFunction {
 private:
  const uint32_t _srps_per_table, _total_num_srps, _dim, _sample_size;
  int16_t* _random_bits;
  uint32_t* _hash_indices;

 public:
  SignedRandomProjection(uint32_t input_dim, uint32_t srps_per_table,
                         uint32_t num_tables,
                         uint32_t seed = global_random::nextSeed());

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  uint32_t hashSingleDenseRow(const float* values, uint32_t row) const;

  std::unique_ptr<HashFunction> copyWithNewSeeds() const final {
    return std::make_unique<SignedRandomProjection>(
        /* input_dim= */ _dim, /* srps_per_table= */ _srps_per_table,
        /* num_tables= */ _num_tables);
  }

  std::string getName() const final { return "SRP"; }

  ~SignedRandomProjection() final;
};

}  // namespace thirdai::hashing
