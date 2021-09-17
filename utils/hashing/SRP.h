// #pragma once

#include "HashFunction.h"
#include <cstdint>
#include <vector>

namespace thirdai::utils {

class SparseRandomProjection : public HashFunction {
 private:
  const uint32_t _srps_per_table, _num_tables, _total_num_srps, _dim,
      _sample_size;
  int16_t* _random_bits;
  uint32_t* _hash_indices;

 public:
  SparseRandomProjection(uint32_t input_dim, uint32_t srps_per_table,
                         uint32_t num_tables, uint32_t seed = time(nullptr));

  void hashSparse(uint64_t num_vectors, const uint32_t* const* indices,
                  const float* const* values, const uint32_t* lengths,
                  uint32_t* output) const override;

  void hashDense(uint64_t num_vectors, uint64_t dim, const float* const* values,
                 uint32_t* output) const override;

  ~SparseRandomProjection();

  uint32_t numTables() const override { return _num_tables; }

  uint32_t range() const override { return 1 << _srps_per_table; }
};

}  // namespace thirdai::utils
