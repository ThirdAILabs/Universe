// #pragma once

#include "HashFunction.h"
#include <cstdint>
#include <vector>

namespace thirdai::utils {

class SparseRandomProjection : public HashFunction {
 private:
  const uint32_t _srps_per_table, _total_num_srps, _dim, _sample_size;
  int16_t* _random_bits;
  uint32_t* _hash_indices;

 public:
  SparseRandomProjection(uint32_t input_dim, uint32_t srps_per_table,
                         uint32_t num_tables, uint32_t seed = time(nullptr));

  void hashSingleSparse(const uint32_t* indices, const float* values,
                        uint32_t length, uint32_t* output) const override;

  void hashSingleDense(const float* values, uint32_t dim,
                       uint32_t* output) const override;

  ~SparseRandomProjection();
};

}  // namespace thirdai::utils
